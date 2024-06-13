import argparse, datetime, LIB_FACEPRIVPRES


def main():
	parser = argparse.ArgumentParser("Face Privacy Preserving via Independet Face Traits");
	subparsers = parser.add_subparsers();

	subparser = subparsers.add_parser("FaceTraits", description = "Generate Face Traits");
	subparser.add_argument("--Backbone", default = "MobileFaceNet", type = str, help ="MobileNet, PFLD, MobileFaceNet");
	subparser.add_argument("--Detector", default = "Retinaface", type = str, help = "MTCNN, FaceBoxes, Retinaface");
	subparser.add_argument("--ImagesPath", default = "", type = str, help = "Images Path");
	subparser.add_argument("--Labels", default = "", type = str, help = "File with face labels");
	subparser.add_argument("--LandMarksFile", default = "Landmarks.csv", type = str, help = "CSV comprising the landmarks");
	subparser.add_argument("--CropTraits", action = "store_true", help = "Generate face traits images");
	subparser.add_argument("--Split", default = 0.9, type = float, help = "Split size");
	subparser.add_argument("--MinClassSize", default = 8, type = int, help = "Minimum number of samples inside the class");
	subparser.add_argument("--Verbose", action = "store_true");
	subparser.set_defaults(func = LIB_FACEPRIVPRES.FaceTraits);

	subparser = subparsers.add_parser("TrainTraitModel", description = "Train a model to extract trait features");
	subparser.add_argument("--DatasetPath", type = str, required = True, help = "Dataset path");
	subparser.add_argument("--TraitModel", default = "ViT-32b", type = str, help = "{ViT-32b, ViT-16b}");
	subparser.add_argument("--Trait", required = True, type = str, help = "{BB_Eyes, BB_Nose, BB_Mouth, BB_Face}");
	subparser.add_argument("--Device", type = str, required = False, default = "cuda:0", help = "GPU device");
	subparser.add_argument("--FilePath", default = "./MODELS/", type = str, help = "Model 2 Save - default ModelName_Time");
	subparser.add_argument("--Criterion", default = "CrossEntropyLoss", type = str, help = "{CrossEntropyLoss, MSELoss}");
	subparser.add_argument("--Optimizer", default = "SGD", type = str, help = "{SGD, Adam}");
	subparser.add_argument("--Scheduler", default = "StepLR", type = str, help = "{StepLR, MultiStepLR,LinearLR}");
	subparser.add_argument("--Epochs", default = 3, type = int, help = "Number of Epochs = 3");
	subparser.add_argument("--LR", default = 0.001, type = float, help = "Learing Rate = 0.001");
	subparser.add_argument("--BS", default = 256, type = int, help = "Batch Size = 256");
	subparser.add_argument("--Weights", default = None, help = "{None, 'DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2'}");
	subparser.add_argument("--AccStop", default = 0.95, type = float, help = "Accuracy Stop = 0.95");
	subparser.add_argument("--Split", default = 0.9, type = float, help = "Split size = 0.8");
	subparser.add_argument("--SampleSize", default = 0, type = int, help = "Sample size of ViT SCALE");	
	subparser.add_argument("--Augmentation", action = "store_true");
	subparser.add_argument("--ParallelModel", action = "store_true");
	subparser.add_argument("--SettingsFile", type = str, default = "", help = "Read and override settings from yaml file");
	subparser.set_defaults(func = LIB_FACEPRIVPRES.TrainTraitModel);

	subparser = subparsers.add_parser("ExtractFeatures", description = "");
	subparser.add_argument("--LandMarksFile", type = str, required = False, default = "");
	subparser.add_argument("--ModelsPath", type = str, required = False, default = "");
	subparser.add_argument("--SampleSize", default = 0, type = int, help = "Sample size of ViT SCALE");		
	subparser.add_argument("--Weights", default = None, help = "{None, 'DEFAULT', 'IMAGENET1K_V1', 'IMAGENET1K_V2'}");	
	subparser.add_argument("--ImagesPath", type = str, required = True);
	subparser.add_argument("--Device", type = str, required = False, default = "cuda:0", help = "GPU device");
	subparser.add_argument("--ParallelModel", action = "store_true");
	subparser.add_argument("--OverwriteCSV", action = "store_true");
	subparser.set_defaults(func = LIB_FACEPRIVPRES.FeatureExtractorLandMarks);

	subparser = subparsers.add_parser("TrainProbabilisticModel", description = "");
	subparser.add_argument("--CSVFile", required = True, type = str, help = "CSV traits features file");
	subparser.add_argument("--ModelsPath", required = False, default = "./MODELS/", type = str, help = "Probabilistic models path to save");
	subparser.add_argument("--NC", required = False, type = int, default = 1, help = "Number of Components");
	subparser.add_argument("--FeatureTransform", default = "L2_distance", required = False, type = str, help = "{raw, L2_distance}");
	subparser.add_argument("--Verbose", action = "store_true");	
	subparser.set_defaults(func = LIB_FACEPRIVPRES.TrainProbabilisticModel);

	subparser = subparsers.add_parser("TestProbabilisticModel", description = "");
	subparser.add_argument("--SRC_CSV", required = True, type = str, help = "CSV souce files");
	subparser.add_argument("--TRG_CSV", required = True, type = str, help = "CSV target files");
	subparser.add_argument("--ModelsPath", required = True, type = str, help = "Probabilistic models path to save");
	subparser.add_argument("--FeatureTransform", default = "L2_distance", required = False, type = str, help = "{raw, L2_distance}");
	subparser.add_argument("--Verbose", action = "store_true");	
	subparser.set_defaults(func = LIB_FACEPRIVPRES.TestProbabilisticModel);


	Options = parser.parse_args();

	print(str(Options) + "\n");

	Response = Options.func(Options);



if __name__ == "__main__":
	print("\n" + "\033[0;32m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
	main();
	print("\n" + "\033[0;32m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");
