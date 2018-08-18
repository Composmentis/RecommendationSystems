import pandas as pd
import numpy as np
import glob
import sys
import pickle
import requests
#import paramiko
import os
from io import BytesIO
from PIL import Image
from sklearn.neighbors import NearestNeighbors
    # from keras.applications.vgg16 import VGG16
    # from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
#from keras.models import Model
from keras.preprocessing import image
from sklearn.externals import joblib

def openingDataFrameForProduct():
    Feature_DataFrame_List = []
    file_path = "/datafolder/"+"*.csv"
    for DataFrame in sorted(glob.glob(file_path)):
            # print(DataFrame)
            Feature_DataFrame = pd.read_csv(DataFrame, encoding="ISO-8859-1", header=0)
            Feature_DataFrame_temp = Feature_DataFrame.loc[:, ['Product_Id', 'Product_Image']]
            #print(DataFrame.split("\\")[-1].split("_")[0])
            Feature_DataFrame_temp['Category']=DataFrame.split("\\")[-1].split("_")[0]
            Feature_DataFrame_List.append(Feature_DataFrame_temp)
    Feature_DataFrame = pd.concat(Feature_DataFrame_List, axis=0, ignore_index=True)
    Feature_DataFrame.to_csv("Feature_DataFrame.csv")
    return Feature_DataFrame

def openingDataFrame(model_name,category):
    # ssh_client=paramiko.SSHClient()
    # ssh_client.load_system_host_keys()
    # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh_client.connect(hostname="bangvmplcep-comm2.sapient.com",username="root",password="Delta@123")
    # ### FTP client
    # sftp_client = ssh_client.open_sftp()
    # apath = '/root/recommendation/Docker/DataFile/'
    # apattern = category + "_Feature_DataFrame_" + model_name + "*.csv"
    # rawcommand = 'find {path} -name {pattern}'
    # command = rawcommand.format(path=apath, pattern=apattern)
    # stdin, stdout, stderr = ssh_client.exec_command(command)
    # filelist = stdout.read().splitlines()
    # Feature_DataFrame_List = []
    # for filename in sorted(filelist):
    #     print(filename.decode("utf-8"))
    #     Feature_DataFrame = pd.read_csv(sftp_client.open(filename.decode("utf-8")), encoding="ISO-8859-1", header=0)
    #     Feature_DataFrame_temp = Feature_DataFrame.loc[:, ['Product_Id', 'Product_Image']]
    #     Feature_DataFrame_List.append(Feature_DataFrame_temp)
    # Feature_DataFrame = pd.concat(Feature_DataFrame_List, axis=0, ignore_index=True)
    # ssh_client.close()
    # return Feature_DataFrame

#     # Method load and open dataframe which holds features of images extracted from pre-trained model
# # input: Pretrained Model name (ResNet50,VGG16,Xception,VGG16_Last)
# # output: DataFrame
# # print(model_name)
#     try:
#         DataFrame = "/datafile/"+category + "_Feature_DataFrame_" + model_name + ".csv"
#         print(DataFrame)
#     except ValueError:
#         print("model_name :", "Invalid model name. Value Error occured_1")
#     except:
#         print(sys.exc_info()[0], " 1occured")
    #try:
        Feature_DataFrame_List = []
        print("opening data frame")
        file_path = "/datafolder/"+category+ "*.csv"
        for DataFrame in sorted(glob.glob(file_path)):
            print(DataFrame)
            Feature_DataFrame = pd.read_csv(DataFrame, encoding="ISO-8859-1", header=0)
            Feature_DataFrame_temp = Feature_DataFrame.loc[:, ['Product_Id', 'Product_Image']]
            Feature_DataFrame_List.append(Feature_DataFrame_temp)
        Feature_DataFrame = pd.concat(Feature_DataFrame_List, axis=0, ignore_index=True)
        Feature_DataFrame.to_csv("Feature_DataFrame.csv")
        return Feature_DataFrame
    # except ImportError:
    #     print("Invalid file is imported")
    # except:
    #     print(sys.exc_info()[0], " 1occured")



def FeatureArrayForInput(input_file, model_name,model_ResNet):

    # Method  to load and open dataframe which holds features of images extracted from pre-trained model
    # input: Pretrained Model name (ResNet50,VGG16,Xception,VGG16_Last)
    # output: numpy array of features
    #try:
        img = requests.get(input_file)
        img_array = image.img_to_array(Image.open(BytesIO(img.content)))
        # img_array = image.img_to_array(image.load_img(input_file))
        img_array = np.expand_dims(img_array, axis=0)
    # except ImportError:
    #     print("Invalid file is imported")
    # except:
    #     print(sys.exc_info()[0], " 2occured")
    # try:
        # if model_name == "ResNet50":
        #     print("ResNet50")
            # from keras.applications.resnet50 import preprocess_input
            # model_ResNet = ResNet50(weights='imagenet', pooling=max, include_top=False)
            # print("model downloaded")
            # input = Input(shape=(224, 224, 3), name='image_input')
            # x = model_ResNet(input)
            # x = Flatten()(x)
            # img_array = preprocess_input(img_array)
            # Feature = model_ResNet.predict(img_array)
            # model_ResNet = Model(inputs=input, outputs=x)
        Feature = model_ResNet.predict(img_array)
        Feature = Feature.squeeze()
        Feature_Array = Feature.flatten(order="C")
        return Feature_Array.reshape(1,2048)
        # elif model_name == "VGG16":
        #     # print("VGG16")
        #     from keras.applications.vgg16 import preprocess_input, decode_predictions
        #     model_VGG16 = VGG16(weights='imagenet', include_top=False)
        #     img_array = preprocess_input(img_array)
        #     Feature = model_VGG16.predict(img_array)
        #     Feature = Feature.squeeze()
        #     Feature_Array = Feature.flatten(order="C")
        #     # print(Feature_Array.shape)
        #     return Feature_Array
        # elif model_name == "VGG16_Last":
        #     # print("VGG16_Last")
        #     from keras.applications.vgg16 import preprocess_input, decode_predictions
        #     base_model_VGG16 = VGG16(weights='imagenet', include_top=False)
        #     model_Layer_Before_Last_Layer_VGG16 = Model(inputs=base_model_VGG16.input,
        #                                                 outputs=base_model_VGG16.get_layer('block4_pool').output)
        #     img_array = preprocess_input(img_array)
        #     Feature = model_Layer_Before_Last_Layer_VGG16.predict(img_array)
        #     Feature = Feature.squeeze()
        #     Feature_Array = Feature.flatten(order="C")
        #     # print(Feature_Array.shape)
        #     return Feature_Array
        # elif model_name == "Xception":
        #     # print("Xception")
        #     from keras.applications.xception import preprocess_input, decode_predictions
        #     model_Xception_base = Xception(weights='imagenet', pooling='max', include_top=False)
        #     model_Xception = Model(inputs=model_Xception_base.input,
        #                            outputs=model_Xception_base.get_layer('block4_pool').output)
        #     img_array = preprocess_input(img_array)
        #     Feature = model_Xception.predict(img_array)
        #     Feature = Feature.squeeze()
        #     Feature_Array = Feature.flatten(order="C")
        #     # print(Feature_Array.shape)
        #     return Feature_Array
    # except ValueError:
    #     print("model_name :", "Invalid model name. Value Error occured_3")
    # except:
    #     print(sys.exc_info()[0], " 3occured")

def nearestNeighbourforImage(number_of_neighbours, model_name,loaded_dataframe,input_file,product,model_ResNet,category,products_To_Be_Considered):

    # Method to train model and predict nearest images
    # input: input file path, number of neighbours to be consideredPretrained Model name (ResNet50,VGG16,Xception,VGG16_Last)
    # output: unique ids of images
    # Fitting the model
    # print(model_name)
    # try:

        # Identifying feature array of input
        # Feature_Array = openingPickFile(model_name, category)
        #Feature_Data_Frame = openingDataFrame(model_name,category)
        #if product:
            #input_file = Feature_Data_Frame.loc[Feature_Data_Frame['Product_Id'] == product]['Product_Image'].values[0]
        Input_Feature_Array = FeatureArrayForInput(input_file, model_name,model_ResNet)
        if len(products_To_Be_Considered)>0:
            loaded_dataframe = loaded_dataframe[loaded_dataframe['Product_Id'].isin(products_To_Be_Considered)]
            loaded_dataframe.reset_index(inplace=True)
        Feature_list = loaded_dataframe['Extracted_Feature'].values.tolist()
        Neighbor_Model = NearestNeighbors(n_neighbors=number_of_neighbours, algorithm='auto', leaf_size=30,metric='braycurtis')
        Fit = Neighbor_Model.fit(Feature_list)
        Nearest_Images = Neighbor_Model.kneighbors(Input_Feature_Array, number_of_neighbours, return_distance=False)
        output_indexes = Nearest_Images.tolist()
        output_products = []
        output_images = []
        for index in output_indexes[0]:
            # print(index)
            if loaded_dataframe.loc[index, ["Product_Id"]].values[0] not in output_products:
                output_products.append(loaded_dataframe.loc[index, ["Product_Id"]].values[0])
                output_images.append(loaded_dataframe.loc[index, ["Product_Image"]].values[0])
            else:
                continue
        return output_images, output_products
        # Feature_Array = openingPickFile(model_name)
        # # print(Feature_Array.shape)
        # Feature_Data_Frame = openingDataFrame(model_name)
        # Neighbor_Model = NearestNeighbors(n_neighbors=number_of_neighbours, algorithm='auto', leaf_size=30,
        #                                   metric='braycurtis')
        # Fit = Neighbor_Model.fit(Feature_Array)
        # filename = 'sitemap_model.pickle'
        # with open(filename, 'wb') as handle:
        #     pickle.dump(Fit, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # # Identifying feature array of input
        # Input_Feature_Array = FeatureArrayForInput(input_file, model_name)
        # # print("input shape is")
        # # print(Input_Feature_Array.shape)
        # Nearest_Images = Neighbor_Model.kneighbors(Input_Feature_Array, number_of_neighbours, return_distance=False)
        # output_indexes = Nearest_Images.tolist()
        # output_products = []
        # output_images = []
        # for index in output_indexes[0]:
        #     # print(index)
        #     if Feature_Data_Frame.loc[index, ["Product_Id"]].values[0] not in output_products:
        #         output_products.append(Feature_Data_Frame.loc[index, ["Product_Id"]].values[0])
        #         output_images.append(Feature_Data_Frame.loc[index, ["Product_Image"]].values[0])
        #         # print(Feature_Data_Frame.loc[index,["Product_Id"]].values[0])
        #         # print(Feature_Data_Frame.loc[index, ["Product_Image"]].values[0])
        #     else:
        #         continue
        #         # output_images=output_images.remove(0)
        # return output_images, output_products
    # except ValueError:
    #     print("Invalid model name/input file name Value Error occured_4")
    # except (TypeError, ZeroDivisionError):
    #     print("Invalid number of neighbours")
    # except:
    #     print(sys.exc_info()[0], " 4occured")


def openingPickFile(model_name):
    # Method load , create and array and extract the features of the image for which nearest images to be predicted
    #  input: Pretrained Model name (ResNet50,VGG16,Xception,VGG16_Last)
    # output: numpy array of features
    try:
        Feature_Array = []
        file_path = "/datafolder/" +"*.pickle"
        for Feature_Array_name in sorted(glob.glob(file_path)):
            with open(Feature_Array_name, "rb") as Feature_List:
                Feature_Array_temp = pickle.load(Feature_List)
                Feature_Array += Feature_Array_temp
        return np.array(Feature_Array)
    except ValueError:
        print("Invalid model/category name. Value Error occured_2")
    except ImportError:
        print("Invalid file is imported")
    except:
        print(sys.exc_info()[0], " occured")


def recommendation(input_file,loaded_dataframe,model_ResNet,category,products_To_Be_Considered,properties):
    #SSH Client
    # ssh_client =None
    # ssh_client = paramiko.SSHClient()
    # ssh_client.load_system_host_keys()
    # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh_client.connect(hostname="bangvmplcep-comm2.sapient.com", username="root", password="Delta@123")
    # sftp_client = ssh_client.open_sftp()

    number_of_neighbours =properties["image_number_of_recommendation"]
    model_name ="ResNet50"
    output_images =""
    output_products=""
        #print(1)
        #loaded_model_men =pickle.load(sftp_client.open("/root/recommendation/Docker/models/sitemap_men_10151_model.pickle"))
    output_images, output_products = nearestNeighbourforImage(number_of_neighbours, model_name,loaded_dataframe,input_file,None,model_ResNet,category,products_To_Be_Considered)
    output = {"product_Id":output_products,
                "Product_Image_Urls":output_images}
    return output

def recommendationforaProduct(product_id,loaded_model,model_ResNet,category,products_To_Be_Considered):
    #SSH Client
    # ssh_client =None
    # ssh_client = paramiko.SSHClient()
    # ssh_client.load_system_host_keys()
    # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh_client.connect(hostname="bangvmplcep-comm2.sapient.com", username="root", password="Delta@123")
    # sftp_client = ssh_client.open_sftp()
    #loaded_model = None
    number_of_neighbours =10
    model_name ="ResNet50"
    output ={}
    output_images =""
    output_products=""
        #print(1)
        #loaded_model_men =pickle.load(sftp_client.open("/root/recommendation/Docker/models/sitemap_men_10151_model.pickle"))
    output_images, output_products = nearestNeighbourforImage(number_of_neighbours, model_name,
                                                                          loaded_model,None,product_id,model_ResNet,category)
    output = {"product_Id":output_products,
                "Product_Image_Urls":output_images}
    return output


output=recommendation(["input_imageurl"])
# # output = recommendationforaProduct(["productid"])
print(output)

