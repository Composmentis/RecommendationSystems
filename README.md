# RecommendationSystems

# 1) Image Based Recommendation Using Keras pre-trained model

Image based recommendations are suitable for fashion, apperals related product. Most of the modern ecommerce application provides similar product as recommendation using metadata of the product. This metadata of the productmost of the cases does not have detail enough to 100% view on products. Image based recommendation includes minute detail of the products from product images and comes up with visually similar products as recommendation.

It involves following three business process steps
<ul>
<li>a) Identifying visual Preference of the user</li> <li>b) Find list of products which are visually very close to user preference </li>
<li>c) Recommend products from aobve step </li>
</ul>
Above steps could be achieved using following technical steps
<ul>
 <li>a) Gather all the images of the product in the catalog</li><li>b) Pre-process and clean the data</li> <li>c) Load pre-trained CNN model (if pre trained model does not work then we can use transform learning options of pre trained model and train with our own data or build new a CNN model)</li> <li>d) Pass all the processed images of the products and perform feature extraction. Remember CNN has three important block as they are scan and create an representation of image (Convolution and pooling layer), develop genralize understanding of the image(Fully connected layer), Classify image. Here we do not need to classify the image. So we will extract vector representationof image after fully connected layer. And store it (extracted feature of all product images)in in external disk. This can be done offline</li><li>e) Let's assume user visits PDP(product detail page) of the ecommerce site. Then the main image of the product is taken and extracted feature like above step</li><li>f) Now use K nearest Neighbor of Sklearn to find nearest neigbor for image of (step e) from all the images of the catlog (step d). Number of recommendation will be K value</li><li>g) Present visually similar product to user</li>
 </ul>
 
 # 2) Behaviour Based Recommendation
 
 Refer slide 6 and 7 in Machine Learning Based Recommendation System.pptx
 
 Extended features of the Recommendation,
 </ul>
 <li>Visually similar product recommendation Can be extended further. For example user can take an image of the competitor product and search in ecommerce site. This way it could be further extended for image based search</li><li>By including User profile data and Product metadata with collaborative filtering recommendation can be personalized further</li>

