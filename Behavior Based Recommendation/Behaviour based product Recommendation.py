import pickle
import pandas as pd
import tensorrec
import tensorflow as tf
from scipy import sparse
#from sklearn.externals import joblib

def fittingtheModel(user_Feature,product_Feature,interaction):
    model = tensorrec.TensorRec()
    #model.fit(interaction, user_Feature, product_Feature, epochs=3, verbose=True)
    model.fit(interaction, user_Feature, product_Feature, epochs=30,learning_rate=0.01, alpha=0.00001, verbose=True)
    print("fitting is done")
    return model

def predict(user_Feature,product_Feature,model):
    print("prediction started")
    # print(user_Feature.shape)
    # print(product_Feature.shape)
    print("-----")
    predictions = model.predict(user_features=user_Feature,item_features=product_Feature)
    print(predictions.shape)
    print("priediction is done")
    print(predictions)
    return predictions

if __name__ =='__main__':
    with open("User_Features_Sparse_Matrix.pickle", "rb") as user_Feature:
        user_Feature = pickle.load(user_Feature)
    print(user_Feature.shape)
    with open("Product_Feature_Sparse_Matrix.pickle", "rb") as product_Feature:
        product_Feature = pickle.load(product_Feature)
    print(product_Feature.shape)
    with open("Behaviour_Sparse_Matrix.pickle", "rb") as interaction:
        interaction = pickle.load(interaction)
    print(interaction.shape)
    model = fittingtheModel(user_Feature,product_Feature,interaction)
    print("fitting --")
    # filename = 'finalized_model.sav'
    #joblib.dump(model, filename)
    del interaction
    with open("Customer_Products.pickle", "rb") as Customer_Products:
        Customer_Products = pickle.load(Customer_Products)
    Customer_Id = (pd.Series(Customer_Products["Customer"])).values
    with open("product_list.pickle", 'rb') as pl:
        product_list=pickle.load(pl)

    product_Feature = pd.read_csv("product_Feature.csv", encoding="ISO-8859-1", header=0)

    def splitDataFrameIntoSmaller(df, chunkSize=5000):
        listOfDf = list()
        numberChunks = len(df) // chunkSize + 1
        for i in range(numberChunks):
            listOfDf.append(df[i * chunkSize:(i + 1) * chunkSize])
        return listOfDf


    product_Feature_List = splitDataFrameIntoSmaller(product_Feature, 860)
    count = 0
    for product_Feature in product_Feature_List:
        products = product_Feature['Unnamed: 0'].tolist()
        print("len of list")
        print(len(products))
        product_Feature = product_Feature.set_index("Unnamed: 0")
        del product_Feature.index.name
        print("before matrixing")
        print(product_Feature.shape)
        product_Feature = sparse.lil_matrix(product_Feature)
        # if value ==7740:
        #     with open("Customer_Products.pickle", "rb") as Customer_Products:
        #         Customer_Products = pickle.load(Customer_Products)
        #     Customer_Id = (pd.Series(Customer_Products["Customer"])).values
        #     filename = "/home/sakthi10feb88/product_Feature" + str(9) + ".pickle"
        #     with open(filename, "rb") as product_Feature_0:
        #         product_Feature_0 = pickle.load(product_Feature_0)
        #     with open(filename, "rb") as product_Feature_0:
        #         product_Feature_0 = pickle.load(product_Feature_0)
        #     print(product_Feature_0.shape)
        #     predictions = predict(user_Feature, product_Feature_0, model)
        #     filename = "predictions" + str(9) + ".pickle"
        #     with open(filename, 'wb') as handle:
        #         pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     df = pd.DataFrame(predictions, columns=product_list[7740:8598])
        #     df['customer_id'] = Customer_Id
        #     print(df.head(5))
        #     print(len(set(df.iloc[:, 1].values.tolist())))
        #     print(df.tail(5))
        #     filename = "predictions" + str(9) + ".csv"
        # else:
        print(product_Feature.shape)
        predictions = predict(user_Feature,product_Feature,model)
        filename = "predictions" + str(count) + ".pickle"
        with open(filename,'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        df = pd.DataFrame(predictions,columns=products)
        df['customer_id'] = Customer_Id
        print(products)
        print(df.head(5))
        print(len(set(df.iloc[:,1].values.tolist())))
        print(df.tail(5))
        filename = "predictions" + str(count) + ".csv"
        df.to_csv(filename)
        count += 1

rank = model.predict_rank(user_features=user_Feature,item_features=product_Feature)
    # with open("rank.pickle",'wb') as handle:
    #     pickle.dump(rank, handle, protocol=pickle.HIGHEST_PROTOCOL)


