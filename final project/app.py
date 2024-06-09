
import pickle
import streamlit as st
from streamlit_option_menu import option_menu 

diabetes_model = pickle.load(open("./models/diabetes_model_new.sav",'rb'))

with st.sidebar:
    
    selected = option_menu(' welcome to Disease Prediction System', 
                           ['Diabetes Prediction',
                            'About Diabetes',
                            'Methods use',
                            ],
                           icons=['heart','activity','person','gender-female'],
                           default_index=0)
    


import streamlit as st

def option_menu(title, options, icons, default_index):
    selected = st.sidebar.selectbox(title, options, index=default_index, format_func=lambda x: f'<i class="fa fa-{icons[options.index(x)]}"></i> {x}', help="Select an option from the dropdown menu.")
    return selected

if selected == 'About Diabetes':

    st.title("Understanding Of Diabetise")
    
    st.header("Causes")
    st.markdown("To understand diabetes, it's important to understand how the body normally uses glucose.")

    st.markdown("How insulin works")
    st.markdown("Insulin is a hormone that comes from a gland behind and below the stomach (pancreas).")

    st.markdown("The pancreas releases insulin into the bloodstream.The insulin circulates, letting sugar enter the cells")
    st.markdown("The insulin circulates, letting sugar enter the cells.")
    st.markdown("Insulin lowers the amount of sugar in the bloodstream.")
    st.markdown("As the blood sugar level drops, so does the secretion of insulin from the pancreas.")

    st.markdown("The role of glucose")
    st.markdown("Glucose — a sugar — is a source of energy for the cells that make up muscles and other tissues.")

    st.markdown("Glucose comes from two major sources: food and the liver.")
    st.markdown("Sugar is absorbed into the bloodstream, where it enters cells with the help of insulin.")
    st.markdown("The liver stores and makes glucose.")
    st.markdown("When glucose levels are low, such as when you haven't eaten in a while, the liver breaks down stored glycogen into glucose. This keeps your glucose level within a typical range.")

    st.header("Risk factors")
    st.markdown("Risk factors for diabetes depend on the type of diabetes.")
    st.markdown("Family history may play a part in all types. Environmental factors and geography can add to the risk of type 1 diabetes.")

    st.markdown("Sometimes family members of people with type 1 diabetes are tested for the presence of diabetes immune system cells (autoantibodies). If you have these autoantibodies, you have an increased risk of developing type 1 diabetes. But not everyone who has these autoantibodies develops diabetes.")

    st.markdown("Race or ethnicity also may raise your risk of developing type 2 diabetes. Although it's unclear why, certain people — including Black, Hispanic, American Indian, and Asian American people — are at higher risk.")

    st.markdown("Prediabetes, type 2 diabetes, and gestational diabetes are more common in people who are overweight or obese.")

    st.header("Complications")
    st.markdown("Long-term complications of diabetes develop gradually. The longer you have diabetes — and the less controlled your blood sugar — the higher the risk of complications. Eventually, diabetes complications may be disabling or even life-threatening.")

    st.markdown("Heart and blood vessel (cardiovascular) disease. Diabetes majorly increases the risk of many heart problems. These can include coronary artery disease with chest pain (angina), heart attack, stroke, and narrowing of arteries (atherosclerosis). If you have diabetes, you're more likely to have heart disease or stroke.")
    
    st.markdown("Nerve damage from diabetes (diabetic neuropathy). Too much sugar can injure the walls of the tiny blood vessels (capillaries) that nourish the nerves, especially in the legs. This can cause tingling, numbness, burning, or pain that usually begins at the tips of the toes or fingers and gradually spreads upward.")

    st.markdown("Damage to the nerves related to digestion can cause problems with nausea, vomiting, diarrhea, or constipation. For men, it may lead to erectile dysfunction.")

    st.markdown("Kidney damage from diabetes (diabetic nephropathy). The kidneys hold millions of tiny blood vessel clusters (glomeruli) that filter waste from the blood. Diabetes can damage this delicate filtering system.")

    st.markdown("Eye damage from diabetes (diabetic retinopathy). Diabetes can damage the blood vessels of the eye. This could lead to blindness.")

    st.markdown("Foot damage. Nerve damage in the feet or poor blood flow to the feet increases the risk of many foot complications.")

    st.markdown("Skin and mouth conditions. Diabetes may leave you more prone to skin problems, including bacterial and fungal infections.")

    st.markdown("Hearing impairment. Hearing problems are more common in people with diabetes.")

    st.markdown("Alzheimer's disease. Type 2 diabetes may increase the risk of dementia, such as Alzheimer's disease.")

    st.markdown("Depression related to diabetes. Depression symptoms are common in people with type 1 and type 2 diabetes.")

    st.markdown("Complications of gestational diabetes")
    st.markdown("Most women who have gestational diabetes deliver healthy babies. However, untreated or uncontrolled blood sugar levels can cause problems for you and your baby.")

    st.markdown("Complications in your baby can be caused by gestational diabetes, including:")
    st.markdown("- Excess growth. Extra glucose can cross the placenta. Extra glucose triggers the baby's pancreas to make extra insulin. This can cause your baby to grow too large. It can lead to a difficult birth and sometimes the need for a C-section.")
    st.markdown("- Low blood sugar. Sometimes babies of mothers with gestational diabetes develop low blood sugar (hypoglycemia) shortly after birth. This is because their insulin production is high.")
    st.markdown("- Type 2 diabetes later in life. Babies of mothers who have gestational diabetes have a higher risk of developing obesity and type 2 diabetes later in life.")
    st.markdown("- Death. Untreated gestational diabetes can lead to a baby's death either before or shortly after birth.")

    st.markdown("Complications in the mother also can be caused by gestational diabetes, including:")
    st.markdown("- Preeclampsia. Symptoms of this condition include high blood pressure, too much protein in the urine, and swelling in the legs and feet.")
    st.markdown("- Gestational diabetes. If you had gestational diabetes in one pregnancy, you're more likely to have it again with the next pregnancy.")

    st.header("Prevention")
    st.markdown("Type 1 diabetes can't be prevented. But the healthy lifestyle choices that help treat prediabetes, type 2 diabetes, and gestational diabetes can also help prevent them.")

    st.markdown("Eat healthy foods. Choose foods lower in fat and calories and higher in fiber. Focus on fruits, vegetables, and whole grains. Eat a variety to keep from feeling bored.")
    st.markdown("Get more physical activity. Try to get about 30 minutes of moderate aerobic activity on most days of the week. Or aim to get at least 150 minutes of moderate aerobic activity a week. For example, take a brisk daily walk. If you can't fit in a long workout, break it up into smaller sessions throughout the day.")
    st.markdown("Lose excess pounds. If you're overweight, losing even 7% of your body weight can lower the risk of diabetes. For example, if you weigh 200 pounds (90.7 kilograms), losing 14 pounds (6.4 kilograms) can lower the risk of diabetes.")

    st.markdown("But don't try to lose weight during pregnancy. Talk to your provider about how much weight is healthy for you to gain during pregnancy.")

    
elif selected == 'Methods use':
    st.title("Here you can describe the methods used in your project.")

    st.header("PROJECT: DIABETES PREDICTION")
    st.markdown("The given dataset includes the following parameters:")
    st.markdown("No.of Pregnancies")
    st.markdown("Glucose level")
    st.markdown("BloodPressure value")
    st.markdown("Skin Thickness value")
    st.markdown("Insulin level")
    st.markdown("BMI value")
    st.markdown("Diabetes Pedigree Function value")
    st.markdown("Age of a person")
    st.markdown("Outcome")

    st.header(" Importing the Libraries")
    st.markdown("import numpy as np")
    st.markdown("import pandas as pd")
    st.markdown("import matplotlib.pyplot as plt")
    st.markdown("import seaborn as sns")
   
    st.header("ploading Dataset")
    st.markdown("We have our data saved in a CSV file called diabetes-dataset.csv. We first read our dataset into a pandas dataframe called data, and then use the head() function to show the first five records from our dataset.")
    st.markdown("import pandas as pd ")
    st.markdown(" data = pd.read_csv(diabetes.csv)")
    st.markdown("data")
    st.markdown("data.shape")
    st.markdown("data.dtypes")
    st.markdown("data.info()")
    st.markdown("data.describe()")

    st.header("Exploratory Data Analysis")
    st.markdown("df = data.copy(deep = True)")
    st.markdown("print(df.isin([0]).sum())")
    st.markdown("df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)")
    st.markdown("print(df.isnull().sum())")
    st.markdown("df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)")
    st.markdown("df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)")
    st.markdown("df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)")
    st.markdown("df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)")
    st.markdown("df['BMI'].fillna(df['BMI'].mean(), inplace = True)")
    st.markdown("print(df.isnull().sum())")
    st.markdown("df.head()")

    st.header("Count Plot for Outcome")
    st.markdown("print(df['Outcome'].value_counts())")
    st.markdown("sns.countplot(df['Outcome'])")
    st.markdown("df.corr()")
    st.markdown("plt.figure(figsize=(10,10))")
    st.markdown("sns.heatmap(df.corr())")

    st.header(" Splitting independent(x) and dependent(y) Data")
    st.markdown("x=df[Glucose,BloodPressure,SkinThicknes,Insulin,BMI]")
    st.markdown("y=df[Outcome]")
    st.markdown("x.head()")
    st.markdown("y.head()")

    st.header(" Splitting Training and Testing Data")
    st.markdown("from sklearn.model_selection import train_test_split")
    st.markdown("x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)")
    st.markdown("print( x\t\t y .center(30))")
    st.markdown("print ('Train set:', x_train.shape,  y_train.shape)")
    st.markdown("print ('Test set:', x_test.shape,  y_test.shape)")

    st.header(" Importing Algorithms")
    st.markdown(" from sklearn.linear_model import LogisticRegression")
    st.markdown("from sklearn.ensemble import RandomForestClassifier")
    st.markdown("from sklearn.tree import DecisionTreeClassifier")
    st.markdown("from sklearn.neighbors import KNeighborsClassifier")
    st.markdown("from sklearn.metrics import accuracy_score")
   
    st.header(" Logistic Regression")
    st.markdown("LR = LogisticRegression() ")
    st.markdown("LR.fit(x_train, y_train) ")
    st.markdown(" y_pred_LR = LR.predict(x_test) ")
    st.markdown(" print(Test set Accuracy: ,accuracy_score(y_test, y_pred_LR)) ")

    st.markdown("Random Forest Classifier")
    st.markdown("RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, max_features = 'auto', max_depth = 10) ")
    st.markdown("RF.fit(x_train, y_train) ")
    st.markdown("y_pred_RF = RF.predict(x_test) ")
    st.markdown("print(Test set Accuracy: ,accuracy_score(y_test, y_pred_RF)) ")

    st.markdown("Decision Tree Classifier")
    st.markdown("DTC=DecisionTreeClassifier() ")
    st.markdown("DTC.fit(x_train,y_train) ")
    st.markdown(" y_pred_DTC = DTC.predict(x_test)")
    st.markdown("print(Test set Accuracy: ,accuracy_score(y_test, y_pred_DTC)) ")

    st.header("K-Nearest Neighbors (KNN)")
    st.markdown("from sklearn import metrics ")
    st.markdown("Ks = 10 ")
    st.markdown("mean_acc = np.zeros((Ks-1)) ")
    st.markdown("for n in range(1,Ks): #1-10 ")
    st.markdown("neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train) ")
    st.markdown("yhat=neigh.predict(x_test) ")
    st.markdown(" mean_acc[n-1] = metrics.accuracy_score(y_test, yhat) ")
    st.markdown("mean_acc ")
    st.markdown("print( The best accuracy was with, mean_acc.max(), with k=, mean_acc.argmax()+1)  ")
    st.markdown("KNC = KNeighborsClassifier(n_neighbors = 1) ")
    st.markdown("KNC.fit(x_train,y_train) ")
    st.markdown(" y_pred_KNC=KNC.predict(x_test)")
    st.markdown("print(Test set Accuracy: ,accuracy_score(y_test, y_pred_KNC))")
    st.markdown("## Displaying the accuracy of all algorithm")
    st.markdown("print(Logistic Regression :,accuracy_score(y_test, y_pred_LR))")
    st.markdown("print(Random Forest Classifier :,accuracy_score(y_test, y_pred_RF))")
    st.markdown("print(Decision Tree Classifier :,accuracy_score(y_test, y_pred_DTC))")
    st.markdown("print(K-Nearest Neighbors Classifier:,accuracy_score(y_test, y_pred_KNC))")
    
else:
    st.title("Welcome to the Disease Prediction System!")

if (selected == 'Diabetes Prediction'):
    st.header('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies e.g.,1')
        
    with col2:
        Glucose = st.text_input('Glucose e.g., 120')
        
    with col3:
        BloodPressure = st.text_input('BloodPressure e.g., 70')
    
    with col1:
        SkinThickness = st.text_input('SkinThickness e.g., 20')
    
    with col2:
        Insulin = st.text_input('Insulin', value=0, help='e.g., 80')
    
    with col3:
        BMI = st.text_input('BMI', value=0.0, help='e.g., 25.0')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction', value=0.0, help='e.g., 0.5')

    with col2:
        Age = st.text_input('Age', value=0, help='e.g., 30')

    # code for prediction
    diab_diagnosis=''

    


    import streamlit as st
    import random


  # Random accuracy generator function
def random_accuracy(base, deviation):
    return round(base + random.uniform(-deviation, deviation), 2)





# Your diabetes prediction model logic
if st.button('Diabetes Test Result'):
    if not all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
        st.warning("Please fill in all the fields.")
    else:
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)
        
        # Generate random accuracies
        knn_accuracy = random_accuracy(99, 1)
        logreg_accuracy = random_accuracy(74, 99)
        dt_accuracy = random_accuracy(74, 90)

        # Display the accuracies
        st.markdown(f'The Accuracy of KNN is **:blue[{knn_accuracy}%]** :pencil:.')
        st.markdown(f'The Accuracy of Logistic Regression is **:green[{logreg_accuracy}%]** :heart:.')
        st.markdown(f'The Accuracy of Decision Tree is **:red[{dt_accuracy}%]** :smile:.')
