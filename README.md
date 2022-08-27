# SMU MSDS Capstone
Contact information:  
    - Zachary Harris: zharris@mail.smu.edu  
    - Gowtham Katta: gkatta@mail.smu.edu  
    - Joseph Woodall: woodallj@mail.smu.edu  
    - Advisor: Dr. Robert Slater: rslater@mail.smu.edu  

## Working project name: Styl

Executive Summary of Service:
Our project will be a supervised learning classification problem, where we will recommend clothing styles that would enhance the online shopping experience for the user. Our current objective is to take a picture of an article of clothing that they own and recommend an outfit based on that picture. Our project consists of multiple parts:

    - The User:
        - Computer vision algorithm to read the picture of clothing the user uploads (The User).

    - The Stylist:
        - Computer vision algorithm to analyze outfits we find on clothing brands’ advertisements, or datasets of outfits from popular clothing brands (The Stylist).

    - The Recommender:
        - Recommendation algorithm to output an outfit based on The Stylists’ findings and User’s preferences (The Recommender).

For example:
The user would take a picture of black jeans and we would recommend other articles of clothing (such as a blue shirt, black shoes) found in stores based on the picture. 
The recommendation would further be based on fit, color, and user preference. 
The user then has the option to buy the recommended articles of clothing within the app. 

#### Application Flow
Please use the below image as reference for the application's flow: 
<img src = "Capstone Application Flow.png" alt = "Capstone Application Flow Diagram"/>

#### Deployment
We are planning on developing a web/mobile application for the user to interact with to use the recommendation service.

#### To Run
pip install -r requirements.txt  
cd src && run.py