import sys
import pathlib
import pandas as pd
import numpy as np
from PIL import Image
import glob
import pickle
import keras as ks
import requests
import multiprocessing as mp
#from .ImageLoading import XMLImageLoading
from multiprocessing import Pool,Queue
from io import BytesIO
from datetime import datetime
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image

def loadingImage(model,model_name,file):
# Method load images and extract features from pretrained model
# input: Pretrained Model (ResNet50,VGG16,Xception,VGG16_Last)
# input: Pretrained Model_name (ResNet50,VGG16,Xception,VGG16_Last)
# output: DataFrame, numpy array and list
    img = requests.get(file)
    img_array = image.img_to_array(Image.open(BytesIO(img.content)))
    img.close()
    # img_array = np.resize(arr, (224, 224, 3))
    #print(file)
    #print(img_array.shape)
    img_array = np.expand_dims(img_array, axis=0)
    Feature = preProcess_Predict(img_array, model, model_name)
    Feature = Feature.squeeze()
    Feature = Feature.flatten(order="C")
    #print(Feature)
    return Feature



def loadingImage_Feature_Extraction(model,model_name,file_List,product_list,category):
         # Method load images and extract features from pretrained model
         # input: Pretrained Model (ResNet50,VGG16,Xception,VGG16_Last)
         #input: Pretrained Model_name (ResNet50,VGG16,Xception,VGG16_Last)
         # output: DataFrame, numpy array and list
    #print("start method")
    try:
        #file_List = file_List[0:10] #comment this line
        Feature_List = []
        Feature_Product_List =[]
        Image_List =[]
        for file,product in zip(file_List,product_list):
            try:
                exception = False
                Feature = loadingImage(model,model_name,file)
                print(file)
                print(product)
                if exception == False:
                    Feature_List.append(Feature)
                    Feature_Product_List.append(product)
                    Image_List.append(file)
            except:
                exception = True
                continue
        Feature_Map = {"Product_Id":Feature_Product_List,"Feature_Array":Feature_List,"Product_Image":Image_List} # remove 0:10
        Feature_DataFrame = pd.DataFrame(Feature_Map)
        #print(Feature_DataFrame)
        return Feature_DataFrame,Feature_List,Feature_Product_List
    except ValueError:
        print("model_name :","Invalid model name. Value Error occured")
    except ImportError:
        print("Invalid file is imported")
    except:
        print (sys.exc_info()[0]," at Line 89")

def preProcess_Predict(img_array,model,model_name):
#input: Method to preprocess images for training model
#input: image array
# input: Pretrained Model (ResNet50,VGG16,Xception,VGG16_Last)
# input: Pretrained Model_name (ResNet50,VGG16,Xception,VGG16_Last)
#output: Feature array
    if model_name =="ResNet50":
        from keras.applications.resnet50 import preprocess_input
        img_array = preprocess_input(img_array)
        print(img_array)
        #lock.acquire()
        Feature = model.predict(img_array)
        #lock.release()
        return Feature
    elif model_name =="VGG16" or model_name=="VGG16_Last":
        #print(Model_Name)
        from keras.applications.vgg16 import preprocess_input, decode_predictions
        img_array = preprocess_input(img_array)
        Feature = model.predict(img_array)
        return Feature
    elif model_name == "Xception":
        #print(Model_Name)
        from keras.applications.xception import preprocess_input, decode_predictions
        img_array = preprocess_input(img_array)
        Feature = model.predict(img_array)
        return Feature


def pickleCreation(feature_array,file_name,model_name):
         # input: Method to create pickle file of extracted images
         # input: Feature array
         # input: Pretrained Model_name (ResNet50,VGG16,Xception,VGG16_Last)
         # output: Pickle

    print("pickle_creation ",file_name)
    path = sys.path[0]
    path = path +"\\"+model_name
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    file_name=path+"\\"+file_name

    try:
        with open(file_name, 'wb') as handle:
            #print(file_name)
            #print(feature_array)
            pickle.dump(feature_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
            Feature_Array_ResNet50 = feature_array
    except ValueError:
        print("Invalid file name or feature array. Value Error occured")
    except:
        print(sys.exc_info()[0], " at Line 141")

def openPickleFile(file_path):
    with open(file_path, "rb") as cat_product_image_map:
        cat_product_image_map = pickle.load(cat_product_image_map)
        return cat_product_image_map

def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))

def doJob(process_id,file_List,product_List,category,model_name,built_model):

    try:
        Feature_DataFrame_name = category + "_Feature_DataFrame_" + model_name + str(process_id)
        Feature_DataFrame_csv = Feature_DataFrame_name + ".csv"
        Feature_Array_name = category + "_Feature_Array_" + model_name + str(process_id)
        Feature_Array_pickle = Feature_Array_name + ".pickle"
        File_Name_List = category + str(process_id)+"_File_Name_List"
        #print(Feature_Array_pickle)
        Feature_DataFrame, Feature_Array, File_Name_List = loadingImage_Feature_Extraction(built_model,
                                                                                                 model_name, file_List,
                                                                                                product_List, category)
        Feature_DataFrame.to_csv(Feature_DataFrame_csv)
        pickleCreation(Feature_Array, Feature_Array_pickle,model_name)
    except:
        print(sys.exc_info()[0])


if __name__ =='__main__':
    file_path = "product_and_image.csv"
    model_name ="ResNet50"
    #xml_path ="file path"
    #XMLImageLoading = XMLImageLoading(xml_path)
    print(model_name)
    built_model =""
        # ResNet50
    if model_name == "ResNet50":
        #print("1")
        built_model = ResNet50(weights='imagenet', pooling=max, include_top=False)
        #print(built_model)
            # input = Input(shape=(224, 224, 3), name='image_input')
            # x = built_model(input)
            # x = Flatten()(x)
            # built_model = Model(inputs=input, outputs=x)
        # VGG16
    elif model_name == "VGG16" or model_name == "VGG16_Last":
        #print("2")
        built_model = VGG16(weights='imagenet', include_top=False)
        # VGG16_Layer_Before_Last_Layer
    elif model_name == "VGG16_Last":
        #print(3)
        base_model_VGG16 = VGG16(weights='imagenet', include_top=False)
        built_model = Model(inputs=base_model_VGG16.input,
                                                    outputs=base_model_VGG16.get_layer('block4_pool').output)
    # Xception
    elif model_name == "Xception":
        #print(4)
        model_Xception_base = Xception(weights='imagenet', pooling='max', include_top=False)
        built_model = Model(inputs=model_Xception_base.input,
                               outputs=model_Xception_base.get_layer('block4_pool').output)

    product_image = pd.read_csv(file_path,)
    file_List =product_image['Product_Image'].tolist()
    product_List = product_image['Product_Id'].tolist()
    List_product_List = chunks(product_List,1000)
    List_file_List =chunks(file_List,1000)
    jobs=[]
        # lock = t.Lock()
    p = Pool()
        #print(built_model)
    for product_List, file_List in zip(enumerate(List_product_List), enumerate(List_file_List)):
            #if product_List[0]>6 and file_List[0]>6: #Comment the line
        doJob(product_List[0], file_List[1], product_List[1], "Temp", model_name, built_model)



#FeatureExtraction= FeatureExtraction("product_image_map.pickle","ResNet50")
#FeatureExtraction= FeatureExtraction("product_image_map.pickle","VGG16")
#FeatureExtraction= FeatureExtraction("product_image_map.pickle","VGG16_Last")
#FeatureExtraction= FeatureExtraction("product_image_map.pickle","Xception")