/*************************************************************************************
			       DEPARTMENT OF ELECTRICAL AND ELECTRONIC ENGINEERING
					   		     IMPERIAL COLLEGE LONDON 

 				      EE 3.19: Real Time Digital Signal Processing
					       Dr Paul Mitcheson and Daniel Harvey

				        		 PROJECT: Frame Processing

 				            ********* ENHANCE. C **********
							 Shell for speech enhancement 

  		Demonstrates overlap-add frame processing (interrupt driven) on the DSK. 

 *************************************************************************************
 				             By Danny Harvey: 21 July 2006
							 Updated for use on CCS v4 Sept 2010
 ************************************************************************************/
/*
 *	You should modify the code so that a speech enhancement project is built 
 *  on top of this template.
 */
/**************************** Pre-processor statements ******************************/
//  library required when using calloc
#include <stdlib.h>
//  Included so program can make use of DSP/BIOS configuration tool.  
#include "dsp_bios_cfg.h"

/* The file dsk6713.h must be included in every program that uses the BSL.  This 
   example also includes dsk6713_aic23.h because it uses the 
   AIC23 codec module (audio interface). */
#include "dsk6713.h"
#include "dsk6713_aic23.h"

// math library (trig functions)
#include <math.h>

/* Some functions to help with Complex algebra and FFT. */
#include "cmplx.h"      
#include "fft_functions.h"  

// Some functions to help with writing/reading the audio ports when using interrupts.
#include <helper_functions_ISR.h>
#define min(a, b) (((a) < (b)) ? (a) : (b)) 

#define WINCONST 0.85185			/* 0.46/0.54 for Hamming window */
#define FSAMP 8000.0		/* sample frequency, ensure this matches Config for AIC */
#define FFTLEN 256					/* fft length = frame length 256/8000 = 32 ms*/
#define NFREQ (1+FFTLEN/2)			/* number of frequency bins from a real FFT */
#define OVERSAMP 4					/* oversampling ratio (2 or 4) */  
#define FRAMEINC (FFTLEN/OVERSAMP)	/* Frame increment */
#define CIRCBUF (FFTLEN+FRAMEINC)	/* length of I/O buffers */

#define BUFFER_LEN	2.5*FSAMP						//2.5 Second buffer
#define SAMPLING_INTERVAL	FFTLEN/OVERSAMP			//How often data is read from ADC. Should be 64 for 10 second buffer and OVERSAMP of 4
#define FRAME_LEN (BUFFER_LEN/SAMPLING_INTERVAL)	//Should be 312

#define OUTGAIN 16000.0				/* Output gain for DAC */
#define INGAIN  (1.0/16000.0)		/* Input gain for ADC  */
// PI defined here for use in your code 
#define PI 3.141592653589793
#define TFRAME FRAMEINC/FSAMP       /* time between calculation of each frame */


/******************************* Global declarations ********************************/

/* Audio port configuration settings: these values set registers in the AIC23 audio 
   interface to configure it. See TI doc SLWS106D 3-3 to 3-10 for more info. */
DSK6713_AIC23_Config Config = { \
			 /**********************************************************************/
			 /*   REGISTER	            FUNCTION			      SETTINGS         */ 
			 /**********************************************************************/\
    0x0017,  /* 0 LEFTINVOL  Left line input channel volume  0dB                   */\
    0x0017,  /* 1 RIGHTINVOL Right line input channel volume 0dB                   */\
    0x01f9,  /* 2 LEFTHPVOL  Left channel headphone volume   0dB                   */\
    0x01f9,  /* 3 RIGHTHPVOL Right channel headphone volume  0dB                   */\
    0x0011,  /* 4 ANAPATH    Analog audio path control       DAC on, Mic boost 20dB*/\
    0x0000,  /* 5 DIGPATH    Digital audio path control      All Filters off       */\
    0x0000,  /* 6 DPOWERDOWN Power down control              All Hardware on       */\
    0x0043,  /* 7 DIGIF      Digital audio interface format  16 bit                */\
    0x008d,  /* 8 SAMPLERATE Sample rate control        8 KHZ-ensure matches FSAMP */\
    0x0001   /* 9 DIGACT     Digital interface activation    On                    */\
			 /**********************************************************************/
};

// Codec handle:- a variable used to identify audio interface  
DSK6713_AIC23_CodecHandle H_Codec;

float *inbuffer, *outbuffer;   		/* Input/output circular buffers */
float *inframe, *outframe;          /* Input and output frames */

complex *intermediate_frame;
complex *intermediate_frame_cpy;
complex *prevFrame;
complex *nextFrame;
float current_sample;
float prev_sample = 0;
float prev_sample_enh3 = 0;
float *min_noise_est;
float *noise_est;

float *inwin, *outwin;              /* Input and output windows */
float ingain, outgain;				/* ADC and DAC gains */ 
float cpufrac; 						/* Fraction of CPU time used */
volatile int io_ptr=0;              /* Input/ouput pointer for circular buffers */
volatile int frame_ptr=0;           /* Frame pointer */
volatile int interval_ptr=0;
volatile int frame_count=0;

float lamda = 0.1;					//minimum noise threshold
float lamda_enh4_1;
float lamda_enh4_2;
float lamda_enh4_3;
float SNR_Threshold = 1.2;
float NSR_Thresold = 3;

float G;
float alpha;
float alpha_default = 2; 					//noise scaling factor
float alpha_increment = 12;
float time_const = 0.04;
float time_const_enh3 = 0.04;
float time_const_enh4 = 0.08;

double K_pole;
double K_pole_enh3;
double K_pole_enh4;

int processing_enable = 1;
int enhancement1_enable = 1;
int enhancement2_enable = 0;
int enhancement3_enable = 0;
int enhancement4_enable = 0;
int enhancement5_enable = 0;
int enhancement6_enable = 1;
int enhancement7_enable = 0;
int enhancement8_enable = 0;
int enhancement9_enable = 0;



 /******************************* Function prototypes *******************************/
void init_hardware(void);    	/* Initialize codec */ 
void init_HWI(void);            /* Initialize hardware interrupts */
void ISR_AIC(void);             /* Interrupt service routine for codec */
void process_frame(void);       /* Frame processing routine */
           

void basic_processing(void);
void no_processing(void);
void enhancement1(void);
/********************************** Main routine ************************************/
void main()
{      

  	int k, j; // used in various for loops
  	
  
/*  Initialize and zero fill arrays */  

	inbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Input array */
    outbuffer	= (float *) calloc(CIRCBUF, sizeof(float));	/* Output array */
	inframe		= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    outframe	= (float *) calloc(FFTLEN, sizeof(float));	/* Array for processing*/
    inwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Input window */
    outwin		= (float *) calloc(FFTLEN, sizeof(float));	/* Output window */

    intermediate_frame = (complex *) calloc(FFTLEN, sizeof(complex));	/* Processing window */
    intermediate_frame_cpy  = (complex *) calloc(FFTLEN, sizeof(complex));
	prevFrame  = (complex *) calloc(FFTLEN, sizeof(complex));
	nextFrame  = (complex *) calloc(FFTLEN, sizeof(complex));
    noise_est = (float *) calloc(OVERSAMP*FFTLEN, sizeof(float));	//Noise estimate. 2D array
	min_noise_est = (float *) calloc(FFTLEN, sizeof(float));
	/* initialize board and the audio port */
  	init_hardware();
  
  	/* initialize hardware interrupts */
  	init_HWI();    
  
/* initialize algorithm constants */  
                       
  	for (k=0;k<FFTLEN;k++)
	{                           
	inwin[k] = sqrt((1.0-WINCONST*cos(PI*(2*k+1)/FFTLEN))/OVERSAMP);
	outwin[k] = inwin[k]; 
	} 
  	ingain=INGAIN;
  	outgain=OUTGAIN;        

  	//Initialise the M bins
	for (k=0; k<OVERSAMP; k++)
		for (j=0; j<FFTLEN; j++)
			noise_est[k*FFTLEN+k] = 9999999999999999;				

	//Initialise K for enhancement1

	K_pole = exp(-TFRAME/time_const);
	K_pole_enh3 = exp(-TFRAME/time_const_enh3);

  	/* main loop, wait for interrupt */  
  	while(1) 	process_frame();
}
    
/********************************** init_hardware() *********************************/  
void init_hardware()
{
    // Initialize the board support library, must be called first 
    DSK6713_init();
    
    // Start the AIC23 codec using the settings defined above in config 
    H_Codec = DSK6713_AIC23_openCodec(0, &Config);

	/* Function below sets the number of bits in word used by MSBSP (serial port) for 
	receives from AIC23 (audio port). We are using a 32 bit packet containing two 
	16 bit numbers hence 32BIT is set for  receive */
	MCBSP_FSETS(RCR1, RWDLEN1, 32BIT);	

	/* Configures interrupt to activate on each consecutive available 32 bits 
	from Audio port hence an interrupt is generated for each L & R sample pair */	
	MCBSP_FSETS(SPCR1, RINTM, FRM);

	/* These commands do the same thing as above but applied to data transfers to the 
	audio port */
	MCBSP_FSETS(XCR1, XWDLEN1, 32BIT);	
	MCBSP_FSETS(SPCR1, XINTM, FRM);	


}
/********************************** init_HWI() **************************************/ 
void init_HWI(void)
{
	IRQ_globalDisable();			// Globally disables interrupts
	IRQ_nmiEnable();				// Enables the NMI interrupt (used by the debugger)
	IRQ_map(IRQ_EVT_RINT1,4);		// Maps an event to a physical interrupt
	IRQ_enable(IRQ_EVT_RINT1);		// Enables the event
	IRQ_globalEnable();				// Globally enables interrupts

}
        
/******************************** process_frame() ***********************************/  
void process_frame(void)
{
	int k, m; 
	int io_ptr0;   

	/* work out fraction of available CPU time used by algorithm */    
	cpufrac = ((float) (io_ptr & (FRAMEINC - 1)))/FRAMEINC;  

	/* wait until io_ptr is at the start of the current frame */ 	
	while((io_ptr/FRAMEINC) != frame_ptr); 

	/* then increment the framecount (wrapping if required) */ 
	if (++frame_ptr >= (CIRCBUF/FRAMEINC)) frame_ptr=0;
 	
 	/* save a pointer to the position in the I/O buffers (inbuffer/outbuffer) where the 
 	data should be read (inbuffer) and saved (outbuffer) for the purpose of processing */
 	io_ptr0=frame_ptr * FRAMEINC;

	/* copy input data from inbuffer into inframe (starting from the pointer position) */ 

	m=io_ptr0;
    for (k=0;k<FFTLEN;k++)
	{                           
		inframe[k] = inbuffer[m] * inwin[k]; 
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	} 

	/************************* DO PROCESSING OF FRAME  HERE **************************/

	if (enhancement8_enable == 1)
	{
		for (k=0;k<FFTLEN;k++)	
		{
			nextFrame[k].r = inframe[k];
			nextFrame[k].i = 0; 
		}
	}
	else 
	{
		for (k=0;k<FFTLEN;k++)										//TODO: Add this to the for-loop above once all this is finished.
		{                           
			intermediate_frame[k].r = inframe[k];
			intermediate_frame[k].i = 0; 
		}
	}

	//Note: Processing is only done every time a frame is completely grabbed from the ADC.  
	if (processing_enable == 1)
	{
		basic_processing();											//Where processing takes place
	}
	else
	{
		no_processing();											//No processing takes place
	}


	/********************************************************************************/

    /* multiply outframe by output window and overlap-add into output buffer */  
                           
	m=io_ptr0;
    
    for (k=0;k<(FFTLEN-FRAMEINC);k++) 
	{    										/* this loop adds into outbuffer */                       
	  	outbuffer[m] = outbuffer[m]+outframe[k]*outwin[k];   
		if (++m >= CIRCBUF) m=0; /* wrap if required */
	}         
    for (;k<FFTLEN;k++) 
	{                           
		outbuffer[m] = outframe[k]*outwin[k];   /* this loop over-writes outbuffer */        
	    m++;
	}	                                   
}        
/*************************** INTERRUPT SERVICE ROUTINE  *****************************/

// Map this to the appropriate interrupt in the CDB file
   
void ISR_AIC(void)
{       
	short sample;
	/* Read and write the ADC and DAC using inbuffer and outbuffer */

	sample = mono_read_16Bit();
	inbuffer[io_ptr] = ((float)sample)*ingain;
		/* write new output data */
	mono_write_16Bit((int)(outbuffer[io_ptr]*outgain)); 

	/* update io_ptr and check for buffer wraparound */    

	if (++io_ptr >= CIRCBUF) io_ptr=0;
}

/************************************************************************************/

void basic_processing(void)
{
	int k, l;
	if (enhancement8_enable != 1)									//n-point FFT
	{
		fft(FFTLEN, intermediate_frame);									
	}
	else
	{
		fft(FFTLEN, nextFrame);										//Perform a delay on the operating buffer to create a next frame buffer and a previous frame buffer
	}

	for (k=0; k<FFTLEN; k++)										//Find the noise (minimum of fft spectrum). Note: 2nd half of FFT is conjugate of first half
	{
		current_sample = cabs(intermediate_frame[k]);

		if (enhancement8_enable == 1)
		{
			intermediate_frame_cpy[k].r = intermediate_frame[k].r;
			intermediate_frame_cpy[k].i = intermediate_frame[k].i;
		}

		if (enhancement1_enable == 1 && enhancement2_enable == 0)	//Enhancement 1&2 - Low pass filter of input
		{
			current_sample = (1 - K_pole)*current_sample + K_pole*prev_sample;	
			prev_sample = current_sample;
		}

		if (enhancement2_enable == 1)
		{
			current_sample = sqrt((1 - K_pole)*current_sample*current_sample + K_pole*prev_sample*prev_sample);	
			prev_sample = current_sample;
		}

		if (current_sample < noise_est[(interval_ptr*FFTLEN)+k])
		{
			noise_est[interval_ptr*FFTLEN+k] = current_sample;
		}
	}

	prev_sample_enh3 = 0;
	if (enhancement3_enable == 1)									//Enhancement 3 - Low pass filter of noise_est
	{
		for (k=0; k<FFTLEN; k++)
		{
			min_noise_est[k] = (1 - K_pole_enh3)*min_noise_est[k] + K_pole_enh3*prev_sample_enh3;	
			prev_sample_enh3 = min_noise_est[k];
		}
	}

	for (k=0; k<FFTLEN; k++)										//Noise subtraction	
	{
		if (enhancement6_enable == 1)								//Enhancement 6 - Increase alpha scale value
		{
			if (SNR_Threshold > (cabs(intermediate_frame[k])/min_noise_est[k]))
			{
				alpha = alpha_default + alpha_increment; 			
			}
		}
		else
		{
			alpha = alpha_default;
		}

		if(enhancement4_enable == (1||2||3||4))						//Enhancement 4 - 
		{
			current_sample = (1 - K_pole_enh4)*current_sample + K_pole_enh4*prev_sample;	
			prev_sample = current_sample;

			if(enhancement4_enable == 1)
			{
				G = 1 - (alpha*min_noise_est[k])/cabs(intermediate_frame[k]);
				lamda_enh4_1 = (lamda*alpha*min_noise_est[k])/cabs(intermediate_frame[k]);
				if(G < lamda_enh4_1)
				{
					G = lamda_enh4_1;
				}
			}

			else if(enhancement4_enable == 2)
			{
				G = 1 - (alpha*min_noise_est[k])/cabs(intermediate_frame[k]);
				lamda_enh4_2 = lamda*current_sample/cabs(intermediate_frame[k]);
				if(G < lamda_enh4_2)
				{
					G = lamda_enh4_2;
				}
			}

			else if(enhancement4_enable == 3)
			{
				G = 1 - (alpha*min_noise_est[k])/current_sample;
				lamda_enh4_3 = (lamda*alpha*min_noise_est[k])/current_sample;
				if(G < lamda_enh4_3)
				{
					G = lamda_enh4_3;
				}
			}

			else if(enhancement4_enable == 4)
			{
				G = 1 - (alpha*min_noise_est[k])/current_sample;
				if(G < lamda)
				{
					G = lamda;
				}
			}					
		}
		else if(enhancement5_enable == 1) 
		{
			G = sqrt(1 - (alpha*min_noise_est[k])*(alpha*min_noise_est[k])/(cabs(intermediate_frame[k])*(cabs(intermediate_frame[k]))));			//Enhancement 5
		}
		else 
		{
			G = 1 - (alpha*min_noise_est[k])/cabs(intermediate_frame[k]);
		}

		if (enhancement8_enable == 1)
		{
			if (min_noise_est[k]/cabs(intermediate_frame[k]) > NSR_Thresold)
			{
				intermediate_frame[k].r = min(prevFrame[k].r,min(intermediate_frame[k].r, nextFrame[k].r));
				intermediate_frame[k].i = min(prevFrame[k].i,min(intermediate_frame[k].i, nextFrame[k].i));
			}
		} 
		else 
		{
			if (G < lamda)
			{
				G = lamda;
			}
			intermediate_frame[k].r 		 = G*intermediate_frame[k].r;
			intermediate_frame[k].i 		 = G*intermediate_frame[k].i;
			//intermediate_frame[FFTLEN-k] = conjg(intermediate_frame[k]);
		}
	}


	ifft(FFTLEN, intermediate_frame);

	//output									
    for (k=0;k<FFTLEN;k++)
	{                           
		outframe[k] = intermediate_frame[k].r;
	} 
	if (enhancement8_enable == 1)
	{
		for (k=0;k<FFTLEN;k++)
		{
			prevFrame[k].r = intermediate_frame_cpy[k].r;
			prevFrame[k].i = intermediate_frame_cpy[k].i;
			intermediate_frame[k].r = nextFrame[k].r;
			intermediate_frame[k].i = nextFrame[k].i;
		}

	}
		
	//Wraparound and comparison of the 4x M bins
	if (frame_count++ > FRAME_LEN)
	{
		frame_count = 0;
		if (interval_ptr++ > OVERSAMP)
		{
			interval_ptr = 0;
		}

		//Reset new bin
		for (k=0; k<FFTLEN; k++)
  			noise_est[interval_ptr*FFTLEN+k] = 9999999999999999;

  		//initialise min_noise_est
  		for (k=0; k<FFTLEN; k++)
		{
			min_noise_est[k] = noise_est[k];			//noise_est[0][k]
		}

		//Compare M_bins
		for (k=0; k<FFTLEN; k++)
		{			
			for (l=0; l<OVERSAMP; l++)
			{
				min_noise_est[k] = min(min_noise_est[k], noise_est[l*FFTLEN+k]);
			}
		}
	}
}

void no_processing(void)
{
	int k;
	for (k=0;k<FFTLEN;k++)
	{                           
		outframe[k] = inframe[k];/* copy input straight into output */ 
	} 
}
