This code implements some tools to implement the technique proposed en the paper https://arxiv.org/pdf/1804.01527.pdf. The technique is applied to offline handwriting recognition tasks over small datasets.

Briefly, the method consists on training a model over a bigger dataset than the target one, e.g. IAM database and after that retraining the model with the small target set one.

Requirements:

	- python 3.6.5
	- Libraries: 
		- TensorFlow 1.5
		- NumPy
		- HDF5 for python (h5py)
		- OpenCV
		- PIL

Steps:

	1º- Train the model with IAM database.
	
		a) Download the database from http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database. You'll need to be registered in the website.
		
		b) Place the folder "lines" and the file "lines.txt" inside the folder "Projects/IAM_lines/Database". 
		
		c) Execute script "create_IAM_lines_dataset.py" in the folder "Projets/IAM_lines/" to prepare the data included in each dataset in "Projets/IAM_lines/Sets" folder. Files in format hdf5 and csv will be created. This files are fed into the framework to train the models. You can edit the "txt" files in this folder indicating the names of the images in each dataset and after that, execute the script.
		
			c.1) Change the current directory to Projects/IAM_lines.
			c.2) In a shell, type: python3 create_IAM_lines_dataset.py
			
		d) Train the model executing the script "Structure_006.py" in the folder "Projets/IAM_lines/Structure_006".
		
			d.1) Change the current directory to "Projets/IAM_lines/Structure_006"
			d.2) Edit the main module in the file "Structure_006.py" by uncommenting the lines in which trainset.h5, testset.h5 and validationset1.h5 are fixed as the training, test and validation sets respectively. Comment the three lines in which demoset appears. 
			     In the main module of this file you can also edit some parameters such as learning rate, total number of epochs, batch size ...
			d.3) Execute the training by typing: python3 Structure_006.py
        			
        			
         e) For each training, a new folder named train-{DATE} will be created containing a file with logs, a folder with the models saved each "num_epochs_before_validation" fixed in the main module of Structure_006.py and TensorBoard files with summaries.
         
             While training you can visualize the progress by changing the current directory to the new folder "train-{DATE}" and typing in a shell:
                 
                 tensorboard --logdir=./TensorBoard_files
                 
             After that you can open a browser and search the URL showed in the shell after executing the command above.
            
         d) When the train is completed, you can find the model in the folder "train-{DATE}/models". This folder contains a file named "checkpoint" where the last epoch is recorded, and three files of each saved epoch: "model-X.data-00000-of-00001", "model-X.index" and "model-X.meta". X is the number of epoch. 
         
    2º Perform transfer learning to train the model in Washington or Parzival databases.
        
        These are the steps for the Parzival databse in the case of 350 images in the training set. Washington is similar.
                
        a) Download the database from http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/parzival-database. You'll need to be registered in the website.
        
        b) Uncompress the downloaded file and place the folder "parzivaldb-v1.0" in the folder "Projects/Parzival/Database/".
        
        c) Execute script "Projets/Parzival/create_Parzival_lines_dataset.py" to prepare the data as in the IAM case. The folder Projects/Parzival/Sets contains .txt files with the name of each image in each dataset.
        
            d.1) Change the current directory to Projects/Parzival. 
            d.2) In a shell, type: python3 create_Parzival_lines_dataset.py
            
        d) Train the model with the parameters initialized from the IAM training.
        
            d.1) Copy the checkpoint file and the three files corresponding to the chosen epoch (see step 1º d ) inside the folder "Projects/Parzival/Structure_006_TL/model_for_transfer/
            
            d.2) Change the current directory to "Projets/Parzival/Structure_006_TL"
            
            d.3) Edit the main module in the file "Structure_006_TL.py" by uncommenting the lines in which train.h5, test.h5 and valid.h5 are fixed as the training, test and validation sets respectively. Comment the three lines in which demoset appears. 
			     In the main module of this file you can also edit some parameters such as learning rate, total number of epochs, batch size ...
			     Notice that the variable "transferFLAG" is set to True.
			     
		   d.4) Execute the training by typing: python3 Structure_006_TL.py
		   
        		   You can visualize the progress in Tensorboard following the steps in 1º e)   
			     
			     
			     
			     
			     

            
        