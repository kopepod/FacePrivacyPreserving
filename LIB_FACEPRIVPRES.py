## v20240612 R. Leyva

from __future__ import division
import sys, argparse, glob, time, pandas, torch, os, cv2, numpy, tqdm, subprocess, yaml, torchvision, copy, datetime, pickle, gc
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import mixture

# DETECTOR [start]
sys.path.append("./DETECTOR/")
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from MTCNN import detect_faces
from utils.align_trans import get_reference_facial_points, warp_and_crop_face
# DETECTOR [end]

numpy.format_float_positional(1e-07)

torch.backends.cudnn.enabled = False
#torch.backends.cudnn.benchmark = True 
#torch.backends.cudnn.deterministic = True

'''

Landmarks

'''

def load_model(Options):
	map_location = lambda storage, loc: storage.cuda() if torch.cuda.is_available() else "cpu";
	match Options.Backbone:
		case "MobileNet":
			model = MobileNet_GDConv(136)
			model = torch.nn.DataParallel(model)
			# download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
			checkpoint = torch.load("./DETECTOR/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar", map_location=map_location)
			print("Use MobileNet as backbone")
		case "PFLD":
			model = PFLDInference() 
			# download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
			checkpoint = torch.load("./DETECTOR/checkpoint/pfld_model_best.pth.tar", map_location=map_location)
			print("Use PFLD as backbone") 
		case "MobileFaceNet":
			model = MobileFaceNet([112, 112],136)   
			# download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
			checkpoint = torch.load("./DETECTOR/checkpoint/mobilefacenet_model_best.pth.tar", map_location=map_location)      
			print("Use MobileFaceNet as backbone")         
		case _ :
			print("Error: not suppored backbone")    
	model.load_state_dict(checkpoint["state_dict"])
	return model

def FaceTraits(Options):

	mean = numpy.asarray([ 0.485, 0.456, 0.406 ]);
	std = numpy.asarray([ 0.229, 0.224, 0.225 ]);
	crop_size = 112;
	scale = crop_size / 112.0;
	reference = get_reference_facial_points(default_square = True) * scale;

	out_size = 224 if Options.Backbone == "MobileNet" else 112;
        
	model = load_model(Options);
	model = model.eval();

	DF = pandas.read_csv(Options.Labels);
	
	DF = DF.assign(RightEye = "");
	DF = DF.assign(LeftEye = "");
	DF = DF.assign(Nose = "");
	DF = DF.assign(RightMouth = "");
	DF = DF.assign(LeftMouth = "");
	DF = DF.assign(BB_Face = "");
   
	print("\nExtracting landmarks ... \n");
  
	for row_idx, row in (pbar := tqdm.tqdm(DF.iterrows(), total = len(DF)) ):
		File = row.File;
		pbar.set_description("Processing file: %s "%(File))
		img = cv2.imread(Options.ImagesPath + File);
		org_img = Image.open(Options.ImagesPath + File);
		height,width,_ = img.shape;
		match Options.Detector:
			case "MTCNN":
				image = Image.open(Options.ImagesPath + File);
				faces, landmarks = detect_faces(image);
			case "FaceBoxes":
				face_boxes = FaceBoxes()
				faces = face_boxes(img)
			case "Retinaface":
				retinaface=Retinaface.Retinaface()    
				faces = retinaface(img)
			case _ :
				print("Error: not suppored detector")        
			
		ratio = 0
		if len(faces)==0:
			if Options.Verbose: print("No detected face");
			continue
		for k, face in enumerate(faces): 
			if face[4] < 0.9: # remove low confidence detection
				continue
			x1, y1, x2, y2, _ = face;
			w = x2 - x1 + 1
			h = y2 - y1 + 1
			size = int(min([w, h])*1.2)
			cx = x1 + w//2
			cy = y1 + h//2
			x1 = cx - size//2
			x2 = x1 + size
			y1 = cy - size//2
			y2 = y1 + size

			dx = max(0, -x1)
			dy = max(0, -y1)
			x1 = max(0, x1)
			y1 = max(0, y1)

			edx = max(0, x2 - width)
			edy = max(0, y2 - height)
			x2 = min(width, x2)
			y2 = min(height, y2)
			new_bbox = list(map(int, [x1, x2, y1, y2]))
			new_bbox = BBox(new_bbox)
			cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
			if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
				cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
			cropped_face = cv2.resize(cropped, (out_size, out_size))

			if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
				continue
			test_face = cropped_face.copy()
			test_face = test_face/255.0
			if Options.Backbone=="MobileNet":
				test_face = (test_face-mean)/std
			test_face = test_face.transpose((2, 0, 1))
			test_face = test_face.reshape((1,) + test_face.shape)
			x_input = torch.from_numpy(test_face).float()
			x_input= torch.autograd.Variable(x_input)
			start = time.time()
			if Options.Backbone=="MobileFaceNet":
				landmark = model(x_input)[0].cpu().data.numpy()
			else:
				landmark = model(x_input).cpu().data.numpy()
			end = time.time()
			landmark = landmark.reshape(-1,2)
			landmark = new_bbox.reprojectLandmark(landmark)
			img = drawLandmark_multiple(img, new_bbox, landmark)
			# crop and aligned the face
			lefteye_x=0
			lefteye_y=0
			for i in range(36,42):
				lefteye_x += landmark[i][0]
				lefteye_y += landmark[i][1]
			lefteye_x = lefteye_x/6
			lefteye_y = lefteye_y/6
			lefteye = [lefteye_x,lefteye_y]

			righteye_x = 0
			righteye_y = 0
			for i in range(42,48):
				righteye_x += landmark[i][0]
				righteye_y += landmark[i][1]
			righteye_x = righteye_x/6
			righteye_y = righteye_y/6
			righteye = [righteye_x,righteye_y]

			nose = landmark[33]
			leftmouth = landmark[48]
			rightmouth = landmark[54]
			facial5points = [righteye,lefteye,nose,rightmouth,leftmouth]
			warped_face = warp_and_crop_face(numpy.array(org_img), facial5points, reference, crop_size=(crop_size, crop_size))
			img_warped = Image.fromarray(warped_face)

			DF.loc[row_idx, "RightEye"] = str(list(map(round, list(righteye))));
			DF.loc[row_idx, "LeftEye"] = str(list(map(round, list(lefteye))));
			DF.loc[row_idx, "Nose"] = str(list(map(round, list(nose))));
			DF.loc[row_idx, "RightMouth"] = str(list(map(round, list(rightmouth))));
			DF.loc[row_idx, "LeftMouth"] = str(list(map(round, list(leftmouth))));
			DF.loc[row_idx, "BB_Face"] = str(list(map(round, [x1,y1,x2,y2] )));
	
	
	DF = GetBoundingBoxes(DF, Options.Verbose);

	if Options.CropTraits:
		print("\nCropping face traits ... \n");
		if Options.Split > 0:
			print("\nSplitting dataset ... \n");
			DFa, DFb = SplitDataset(DF, Options);
			CropTraits(DFa, Options, Stage = "Train");
			CropTraits(DFb, Options, Stage = "Test");
		else:
			CropTraits(DF, Options);


	print("\nWritting %s ...\n"%(Options.LandMarksFile));
	DF.to_csv(Options.LandMarksFile, index = False)
	

def SplitDataset(DF, Options):
	DFtrain = pandas.DataFrame();
	DFtest = pandas.DataFrame();
	for cls in DF["ID"].unique():
		sDF = DF[DF["ID"] == cls];
		if len(sDF) < Options.MinClassSize:
			continue
		sDFa, sDFb = train_test_split(sDF, train_size = Options.Split, random_state = 42);
		DFtrain = pandas.concat([DFtrain,sDFa]);
		DFtest = pandas.concat([DFtest,sDFb]);
	#		
	return DFtrain, DFtest

	
def getMouth(LeftMouth, RightMouth, WideMouth):
	try:
		L_mouth = numpy.fromstring(str(LeftMouth[1:-1]), dtype = numpy.float32, sep = ",");
		R_mouth = numpy.fromstring(str(RightMouth[1:-1]), dtype = numpy.float32, sep = ",");
		Mouth_distance = numpy.linalg.norm(L_mouth-R_mouth);
		BB = numpy.concatenate(( L_mouth - Mouth_distance * WideMouth, R_mouth + Mouth_distance * WideMouth) );
		BB = list(map(round, BB));
	except:
		BB = float("nan"); # not face
	return BB;

def getEyes(LeftEye, RightEye, WideEyes):
	try:
		L_eye = numpy.fromstring(str(LeftEye[1:-1]), dtype = numpy.float32, sep = ",");
		R_eye = numpy.fromstring(str(RightEye[1:-1]), dtype = numpy.float32, sep = ",");
		Eyes_distance = numpy.linalg.norm(L_eye-R_eye);
		BB = numpy.concatenate(( L_eye - Eyes_distance * WideEyes, R_eye + Eyes_distance * WideEyes) );
		BB = list(map(round, BB));
	except:
		BB = float("nan"); # not face
	return BB;
	
def getNose(Nose, LeftEye, RightEye, WideNose):
	try:
		Nose = numpy.fromstring(str(Nose[1:-1]), dtype = numpy.float32, sep = ",");
		L_eye = numpy.fromstring(str(LeftEye[1:-1]), dtype = numpy.float32, sep = ",");
		R_eye = numpy.fromstring(str(RightEye[1:-1]), dtype = numpy.float32, sep = ",");
		Eyes_distance = numpy.linalg.norm(L_eye-R_eye);
		x0 = Nose[0] - Eyes_distance/2;
		y0 = Nose[1] - Eyes_distance/2 * WideNose[0];
		x1 = Nose[0] + Eyes_distance/2;
		y1 = Nose[1] + Eyes_distance/2 * WideNose[1];
		BB = [x0, y0, x1, y1];
		BB = list(map(round, BB));
	except:
		BB = float("nan"); # not face
	return BB;


def GetBoundingBoxes(DF, Verbose = False):
	WideEyes = numpy.asarray([0.5, 0.25]); # [wide span on x, wide span on y]
	WideNose = numpy.asarray([0.9, 0.25]); # [gap before the nose, gap after the nose times the eyes separation]
	WideMouth = numpy.asarray([0.3, 0.2]); # [wide span on x, wide span on y]
	DF = DF.assign(BB_Eyes = "");
	DF = DF.assign(BB_Nose = "");
	DF = DF.assign(BB_Mouth = "");
	#
	if Verbose: print("\nComputing eyes bounding boxes ... ");
	DF.BB_Eyes = DF.apply(lambda x : getEyes(x.LeftEye, x.RightEye, WideEyes), axis = 1 );
	if Verbose: print("\nComputing nose bounding boxes ... ");
	DF.BB_Nose = DF.apply(lambda x : getNose(x.Nose, x.LeftEye, x.RightEye, WideNose), axis = 1 );
	if Verbose: print("\nComputing mouth bounding boxes ... ");
	DF.BB_Mouth = DF.apply(lambda x : getMouth(x.LeftMouth, x.RightMouth, WideMouth), axis = 1 );	
	#
	DF = DF.dropna();
	return DF;


def CropTraits(DF, Options, Stage = "Train"):

	TPath = Options.ImagesPath + "Traits/" + Stage + "/";

	if os.path.exists(TPath):
		os.system("rm -rf %s" % (TPath));
	
	for _, row in (pbar := tqdm.tqdm(DF.iterrows(), total = len(DF)) ):
		pbar.set_description("Processing file: %s "%(row["File"]))
		I = Image.open(Options.ImagesPath + row.File);
		for Trait in ["Face", "Eyes", "Nose", "Mouth"]:
			try:
				BB = numpy.fromstring(str(row["BB_" + Trait])[1:-1], sep = ",", dtype = numpy.int32);
				Ic = I.crop(tuple(BB));
				TraitPath = TPath + Trait + "/" + str(row.ID) + "/";
				if not os.path.exists(TraitPath):
					os.system("mkdir -p %s" %(TraitPath) );
				#	
				Ic.save(TraitPath + row.File);
				Ic.close()
			except:
				continue
		I.close()

	os.system("mv %s %s" %(Options.ImagesPath + "Traits/", "./DATASETS/") )


'''

Model Banks

'''

def loadSettingsYAML(File):
	class Options: pass
	with open(File) as fid:
		docs = yaml.load_all(fid, Loader = yaml.FullLoader)
		for doc in docs:
			for k, v in doc.items():
				setattr(Options, k, v)
	return Options;

Options = loadSettingsYAML("Settings.yaml"); # Future release, now parser for experimentation purposes

def TrainTraitModel(Options):
	Dataset = GenerateDataset(Options, "Train");
	Model = SelectModel(Options, len(Dataset.get("ClassNames")));
	device = torch.device(Options.Device);
	mssg = "Device: %s" %(device);
	print("\n" + "\033[1;33m" + mssg + "\033[0m" + "\n");
				
	if Options.ParallelModel:
		Model = torch.nn.DataParallel(Model);
		print("\n" + "\033[1;33m" + "Using multi-GPU support" + "\033[0m" + "\n");

	Model.to(device)

	Optimizer = Options.Optimizer
	Scheduler = Options.Scheduler
	Epochs = Options.Epochs
	LR = Options.LR
	AccStop = Options.AccStop
	Criterion = Options.Criterion	

	Criteria = { 
		"BCELoss" : torch.nn.BCELoss(),
		"CrossEntropyLoss": torch.nn.CrossEntropyLoss(),
		"MSELoss": torch.nn.MSELoss() 
	}
		
	Criterion = Criteria.get(Criterion);	

	Optimizers = { 
		"SGD": torch.optim.SGD(Model.parameters(), lr = LR, momentum = 0.9),
		"SGD2": torch.optim.SGD(Model.parameters(), lr = LR, momentum = 0.9, weight_decay = 5e-5, nesterov = True ),
		"Adam": torch.optim.Adam(Model.parameters(), betas=[0.1,0.99], lr = LR)
	}
	
	Optimizer = Optimizers.get(Optimizer);	

	Schedulers = { 
		"LinearLR": torch.optim.lr_scheduler.LinearLR(Optimizer, start_factor = 1.0, end_factor = 0.1, total_iters = int(Epochs)),		
		"StepLR": torch.optim.lr_scheduler.StepLR(Optimizer, step_size = 7, gamma = 0.1),
		"MultiStepLR": torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones = [30,80], gamma = 0.1)
	}
	
	Scheduler = Schedulers.get(Scheduler);
	
	DatasetSize = Dataset.get("DatasetSize");
	Dataloaders = Dataset.get("Dataloaders");

	RunTime = - time.time();
	BestModel = copy.deepcopy(Model.state_dict());
	best_acc = 0.0
	for epoch in range(Epochs):
		print("Epoch {}/{}\n".format(epoch, Epochs - 1));
		# Each epoch has a training and validation phase
		for phase in ["Train", "Validate"]:
			if phase == "Train":
				Model.train()  # Set model to training mode
			else:
				Model.eval()   # Set model to evaluate mode
				#
			running_loss = 0.0
			running_corrects = 0
			# Iterate over data.
			for x, y in (pbar := tqdm.tqdm( Dataloaders[phase] , total = len(Dataloaders[phase]) ) ): 
				x = x.to(device);
				y = y.to(device);
				# zero the parameter gradients
				Optimizer.zero_grad();
				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == "Train"):
					#
					y_hat = Model(x);
					# loss
					loss = Criterion(y_hat, y);
					# labels by argmax
					_, y_hat = torch.max(y_hat, 1);
					if phase == "Train":
						#loss.backward(retain_graph = True);
						loss.backward();
						Optimizer.step();
						#Optimizer.zero_grad(); # stalled loss
					# statistics
					running_loss += loss.item() * x.size(0);
					e_acc = y_hat == y.data;
					running_corrects += torch.sum(e_acc);6
					mssg = "GPU[M]: " + getGPUmem() + " LR: %0.4f  Loss: %0.5f eAcc: %0.6f" %(Optimizer.param_groups[0]["lr"], loss.item(), torch.mean(e_acc.float()) );
					pbar.set_description( mssg );			
					if loss.item() > 100:
						print("Check parameters ...")
						return

			if phase == "Train":
				Scheduler.step();
			epoch_loss = running_loss / DatasetSize[phase]
			epoch_acc = running_corrects.double() / DatasetSize[phase];
			print("\n{:} stage \t Loss: {:0.4f} Acc: {:0.4f}\n".format(phase,epoch_loss, epoch_acc));
				# deep copy the model
			if phase == "Validate" and epoch_acc > best_acc:
				best_acc = epoch_acc
				BestModel = copy.deepcopy(Model.state_dict());
			if epoch_acc.detach().cpu().numpy() > AccStop:
				break
		if epoch_acc.detach().cpu().numpy() > AccStop:
			print("Best accuracy stop reached ... finishing training");
			break
	RunTime += time.time();
	print("Training finished in {:0.0f} seconds".format(RunTime));
	print("Best Acc: {:0.4f}".format(best_acc));
	# load best model weights
	Model.load_state_dict(BestModel);

	FileName = Options.FilePath + Options.TraitModel + "__" + Options.Trait + ".pth" ;
		
	print("Saving model : %s" %(FileName));
	#torch.save(Model, FileName);
	
	if Options.ParallelModel:
		print("Saving parallel module")
		torch.save(Model.module.state_dict(), FileName);
	else:
		print("Saving single module")
		torch.save(Model.state_dict(), FileName);
	
	print("\n\n Testing Stage ... \n\n")
	
	auxSplit = Options.Split;
	Options.Split = 0;
	
	Dataset = GenerateDataset(Options, "Test");
	
	Options.Split = auxSplit;
	
	Dataloaders = Dataset.get("Dataloaders");
	test_loader = Dataloaders["Test"];
	DatasetSize = Dataset.get("DatasetSize");
	
	Model.eval();	
	
	with torch.no_grad():
		acc = 0
		for _, (x, y) in tqdm.tqdm(enumerate(test_loader, 0), total = len(test_loader) ):
			y_hat = Model(x.to(device));
			y = y.to(device);
			_, y_hat = torch.max(y_hat, 1);
			y_hat = y_hat.float();
			y = y.float();
			acc += torch.sum(y_hat == y.data);
			#acc += torch.sum( (y_hat > 0.5) == (y > 0) );
						
	acc = acc.detach().cpu().numpy() / DatasetSize["Test"];
	
	print("\nTest accuracy : %0.5f\n" %(acc));
	
	Stats = {
	"Date": str(datetime.datetime.now()), 
	"Model": Options.TraitModel, 
	"RunTime": int(RunTime), 
	"Parameters": count_parameters(Model), 
	"Acc": acc, 
	"Split" : Options.Split, 
	"Settings": str(Options) };

	if os.path.exists("Log.csv"):
		DF = pandas.read_csv("Log.csv");
	else:
		DF = pandas.DataFrame(columns = ["Date", "Model", "RunTime", "Parameters", "Acc", "Split", "Settings"]);
	
	
	DF = pandas.concat([DF, pandas.DataFrame([Stats])], ignore_index = True);

	print("Writing Log.csv ... \n") 

	DF.to_csv("Log.csv", index = False);

def getTransforms(Options):
	data_transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]);	

	if Options.SampleSize == 0:
		match Options.Trait:
			case "Full":
				Options.SampleSize = 224;
			case "BB_Eyes":
				Options.SampleSize = 96;
			case "BB_Nose":
				Options.SampleSize = 96;
			case "BB_Mouth":
				Options.SampleSize = 96;
			case "BB_Face":
				Options.SampleSize = 128;
			case _:
				print("Error : unsupported trait")
				return -1

	data_transforms.transforms.append( torchvision.transforms.Resize(size = (Options.SampleSize, Options.SampleSize)) );

	if hasattr(Options, "Augmentation") and Options.Augmentation:
		data_transforms.transforms.append(torchvision.transforms.RandomHorizontalFlip());

	return data_transforms;

def GenerateDataset(Options, Stage):

	data_transforms = getTransforms(Options);


	if Stage == "Train":
		dataset = torchvision.datasets.ImageFolder(Options.DatasetPath + "train/", data_transforms);
		
		train_idx, validate_idx = train_test_split(list(range(len(dataset))), train_size = Options.Split);
	
		train_dataset = torch.utils.data.Subset(dataset, train_idx);
		validate_dataset = torch.utils.data.Subset(dataset, validate_idx);

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = Options.BS, shuffle = True, num_workers = 4);
		validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size = Options.BS, shuffle = True, num_workers = 4);

		Dataloaders = {"Train": train_loader, "Validate": validate_loader}
		DatasetSize = {"Train": len(train_loader.dataset), "Validate": len(validate_loader.dataset)}
		ClassNames = validate_dataset.dataset.classes
	else:
		dataset = torchvision.datasets.ImageFolder(Options.DatasetPath + "test/", data_transforms);		
		test_loader = torch.utils.data.DataLoader(dataset, batch_size = Options.BS, shuffle = True, num_workers = 4);
		Dataloaders = {"Train": "", "Test": test_loader}
		DatasetSize = {"Train": 0, "Test": len(test_loader.dataset)}
		ClassNames = test_loader.dataset.classes

	print(DatasetSize);
	print("\n");

	Dataset = {
		"Dataloaders" : Dataloaders,
		"DatasetSize" : DatasetSize,
		"ClassNames" : ClassNames
	}
	
	return Dataset
	
def SelectModel(Options, nclasses = 9346, image_size = 224):

	if hasattr(Options, "SampleSize"):
		image_size = Options.SampleSize;

	print("Torchvision version %s" %( torchvision.__version__) );

	ModelName = Options.TraitModel.split("/")[-1]; # filter out path
	ModelName = ModelName.split("_")[0];
	
	print(ModelName)

	match ModelName: 
		case "ViT-16b":
			Model = torchvision.models.vision_transformer.vit_b_16(weights = Options.Weights, num_classes = nclasses, image_size = image_size);
		case "ViT-16L":
			Model = torchvision.models.vision_transformer.vit_l_16(num_classes = nclasses, image_size = image_size);
		case "ViT-32b":
			Model = torchvision.models.vision_transformer.vit_b_32(weights = Options.Weights, image_size = image_size);
			Model.heads.append(torch.nn.Linear(1000, nclasses))
		case "ViT-32L":
			Model = torchvision.models.vision_transformer.vit_l_32(num_classes = nclasses, image_size = image_size);
		case "ViT-14":
			Model = torchvision.models.vision_transformer.vit_h_14(num_classes = nclasses, image_size = image_size);
		case _ :
			print("Not supported model ...")
			return -1;


	return Model

'''

Feature Extractor

'''

def FeatureExtractorLandMarks(Options):

	Models = glob.glob(Options.ModelsPath + "*.pth");
	
	assert len(Models) > 0, "Empty models path";
	
	print(Models);

	DF = pandas.read_csv(Options.LandMarksFile);
		
	print(DF)

	for ModelName in Models:
		print("\nProcessing model %s\n" %(ModelName));
	
		Options.TraitModel = ModelName;
		Options.Trait = ModelName.split("BB_")[-1];
		Options.Trait = Options.Trait.split(".")[0];
		Options.Trait = "BB_" + Options.Trait;

		data_transforms = getTransforms(Options);		
		
		Model = SelectModel(Options, nclasses = 9346);
		Model.load_state_dict(torch.load(ModelName) );

		if Options.ParallelModel:
			Model = torch.nn.DataParallel(Model);
			print("\n" + "\033[1;33m" + "Using multi-GPU support" + "\033[0m" + "\n");

		device = torch.device(Options.Device);
		Model.to(device)	
		Model.eval();
	
		Z = [];
		
		with torch.no_grad():
			for _, row in (pbar := tqdm.tqdm( DF.iterrows() , total = len(DF) ) ): 
				pbar.set_description( "File: %s"  %(Options.ImagesPath + row.File) );
				I = Image.open(Options.ImagesPath + row.File);
				#
				BB = numpy.fromstring(row[Options.Trait][1:-1], sep = ",", dtype = numpy.int32);
				if len(BB) == 0: # no bounding box coming from landmarks file
					Z.append("[]");		
					continue
				try:
					I = I.crop( tuple(BB) ); # wrong bounding box detection
				except:
					Z.append("[]");
					continue
				if 0 in I.size: # zero-sized Image
					Z.append("[]");
					continue
				#			
				z = torch.autograd.Variable(data_transforms(I).unsqueeze(0));
				z = z.to(device);
				z = Model._process_input(z);
				# Expand the class token to the full batch
				batch_class_token = Model.class_token.expand(z.shape[0], -1, -1);
				z = torch.cat([batch_class_token, z], dim = 1);
				z = Model.encoder(z);
				# We're only interested in the representation of the classifier token that we appended at position 0
				z = z[:, 0];
				z = z.squeeze().flatten();
				z = z.cpu().detach().numpy();
				z = numpy.array2string(z, precision = 5, separator =',', suppress_small = True);
				z = z.replace("\n","");
				z = z.replace(" ","");
				Z.append(z);

		DF[Options.Trait + "_Features"] = pandas.Series(Z);

	print(DF)
	
	if Options.OverwriteCSV:
		OutFile = Options.LandMarksFile;
	else:
		OutFile = Options.LandMarksFile.replace(".csv","_Features.csv");

	print("Writing ... %s " %(OutFile) );
	DF.to_csv(OutFile, index = False);


'''

Probabilistic models

'''

def FormatFeatures(DF, Feature):

	for _, row in DF[Feature].items():
		d = row.count(",");
		if d > 0:
			break
	
	X = DF[Feature].apply(lambda x : numpy.fromstring(x[1:-1], sep = ",") if x != "[]" else numpy.zeros(d+1)-1 );
	X = numpy.vstack(X);
	device =  torch.device("cuda" if torch.cuda.is_available() else "cpu");
	print("device: %s \n" % (str(device)));	
	X = torch.from_numpy(X);
	X = X.to(device);
	return X;

def DistanceFeatures(X0, X1, FIllDiag = True, p = 2):
	split_samples = int(500);
	D = numpy.array([], dtype = object); # distance matrix
	#	
	if len(X1) == 0:
		for x in tqdm.tqdm( torch.split(X0, split_samples) , total = round(len(X0) / split_samples)):
			d = torch.cdist(X0, x, p = p); 
			if FIllDiag:
				#d.diagonal().copy_(torch.rand(d.diagonal().size()));
				d.fill_diagonal_(10000); # preventing self distance
			d , _  = torch.sort(d, dim = 0);
			d = torch.mean(d[0:2,:], dim = 0);
			d = d.cpu().detach().numpy();
			D = numpy.append(D,d);
	elif len(X1) < split_samples:
		D = torch.cdist(X0, X1, p = p);
		#D = torch.min(D, dim = 0)[0];
		D , _  = torch.sort(D, dim = 0);
		D = torch.mean(D[0:2,:], dim = 0);
		D = D.cpu().detach().numpy();
	else:
		for x in tqdm.tqdm( torch.split(X1, split_samples), total = round(len(X1) / split_samples) ):
			d = torch.cdist(X0, x, p = p);
			d , _  = torch.sort(d, dim = 0);
			d = torch.mean(d[0:2,:], dim = 0);
			d = d.cpu().detach().numpy();
			D = numpy.append(D, d);
		#
	
	D = numpy.asarray(D, dtype = numpy.float32);
	D = D.flatten();
	#
	return D.reshape(-1, 1);

def ScoreModel(Model, DF, Feature):
	DF = DF[ DF[Feature] != "[]"];
	X = DF[Feature].apply(lambda x : numpy.fromstring(x[1:-1], sep = ",") );
	X = numpy.vstack(X);
	return Model.score_samples(X);

def TrainProbabilisticModel(Options):
	print("\nReading ... %s" %(Options.CSVFile))
	DF = pandas.read_csv(Options.CSVFile); # Source reference
	#
	Features = ["BB_Eyes_Features", "BB_Nose_Features", "BB_Mouth_Features", "BB_Face_Features","BB_Fusion_Features"];
	#
	TPath = "%s%s%s/" %(Options.ModelsPath, "BMM_Models_" , datetimestring());
	
	os.system("mkdir -p  %s" %(TPath));
	
	#
	for f in Features:
		print("Processing [%s] " %(f));
		if f != "BB_Fusion_Features":
			X = FormatFeatures(DF, f);

			match Options.FeatureTransform:
				case "L2_distance":
					X = DistanceFeatures(X, []); # self distance
					gc.collect()
					torch.cuda.empty_cache()
				case _:
					print("No features map, processing raw");
		
		try:
			X_fus = numpy.hstack((X_fus, X.copy() ));
			print("Features fusion ....")
		except:
			X_fus = X.copy();
			print("Initialization features fusion ... ")

		if f == "BB_Fusion_Features":
			X = X_fus;
		
		Model = mixture.BayesianGaussianMixture(n_components = Options.NC, max_iter = 250, covariance_type = "diag", verbose = True);
		Model.fit(X);
		
		FileName = "%s%s%s" %(TPath, f, ".p");
		
		pickle.dump( Model, open( FileName, "wb" ) );
	
	
def TestProbabilisticModel(Options):
	print("\nReading ... %s" %(Options.SRC_CSV))
	DF_src = pandas.read_csv(Options.SRC_CSV); # Source
	#DF_src = DF_src[0:1000];
	print("\nReading ... %s\n" %(Options.TRG_CSV))
	DF_trg = pandas.read_csv(Options.TRG_CSV); # Target
	#DF_trg = DF_trg[0:1000];
	#
	Features = ["BB_Eyes_Features", "BB_Nose_Features", "BB_Mouth_Features", "BB_Face_Features","BB_Fusion_Features"];
	#
	y_trg = {};
	
	#
	for f in Features:
		print("Processing [%s] " %(f));
		if f != "BB_Fusion_Features":
			X_src = FormatFeatures(DF_src, f);
			X_trg = FormatFeatures(DF_trg, f);

			match Options.FeatureTransform:
				case "L2_distance":
					X_trg = DistanceFeatures(X_src, X_trg, FIllDiag = True, p = 2); # target distance
					gc.collect()
					torch.cuda.empty_cache()
				case _:
					print("No features map, processing raw");
		
		try:
			X_trg_fus = numpy.hstack((X_trg_fus, X_trg.copy() ));
			print("Features fusion ....")
		except:
			X_trg_fus = X_trg.copy();
			print("Initialization features fusion ... ")

		if f == "BB_Fusion_Features":
			X_trg = X_trg_fus;
		
		
		Model = pickle.load( open( Options.ModelsPath + f + ".p", "rb" ) );
		y_trg = Model.score_samples(X_trg);
		#
		DF_trg[f.replace("Features","Score")] = y_trg;
		
	OutFile = Options.TRG_CSV.replace(".csv","_Scores.csv");
	
	print("\nWritting %s ... \n"%(OutFile));
		
	DF_trg.to_csv(OutFile, index = False);	

'''

Misc functions

'''


def getGPUmem():
	command = "nvidia-smi --query-gpu=memory.used --format=csv"
	mystr = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:];
	mystr = "".join(mystr);
	mystr = mystr.split(" MiB");
	mystr = "/".join(mystr) 
	return mystr[:-1] + " MiB";


def count_parameters(model):
	parms = sum(p.numel() for p in model.parameters() if p.requires_grad);
	return parms

def datetimestring():
	now = datetime.datetime.now()
	return now.strftime("%Y_%m_%d_%H_%M_%S")	




























'''
EOF
'''



	
