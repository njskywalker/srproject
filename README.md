# Packages
- ipykernel
- numpy
- pandas

# Milestone 1-3: Cleaning Data
- Typecast string price into float ('£2,500.00' to 2500.00)
- Found no missing values from my .unique() and other exploration
- TODO: Pushed Docker image to Hub, then EC2 instantiated it
- TODO: Cleaned images from EC2 attached storage using clean_images.py function

# Milestone 4a: TF-IDF 
- Predicted the price of products using TF-IDF from their descriptions
- Metric was RMSE, and it is huge, over 300k - but to be expected with such little data!

# Milestone 4b: Multiclass LogReg for images
- Predicted product category (most general, e.g. "Home" instead of "Home / Bedrooms / Beds")
  using multiclass logistic regression
- Metric used was accuracy, but also did confusion matrix (precision + sensitivity)