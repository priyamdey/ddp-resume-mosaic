# ddp-resume-mosaic
Testing MosaicML v0.9.0 ddp resume training functionality

Checkout run.sh file for training / resuming.

At surgery time, a new linear head is added to the existing model and the batch size is scaled to 2x. Optimizer, scheduler and loader is reinitialized to reflect the changes. 
All this is done by the algorithm **HeadAdder** inside the head_adder.py file.
