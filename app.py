import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title( "🚦 Road Accident Prediction Using Machine Learning")

df = pd.read_csv("accidentt.csv")
st.success("Dataset Loaded Successfully")
st.write("Total Records:", len(df))
st.write("Total Columns:", len(df.columns))
st.dataframe(df.head())

df.dropna(inplace=True)

label_encoders={}
for col in ["State","District","City","Road_Category","Weather_Condition","Danger_Level"]:
    if col in df.columns:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
        label_encoders[col]=le

features=["Year","State","District","City","Population","Registered_Vehicles","Total_Accidents","Fatal_Accidents","Minor_Injuries","Major_Injuries","Accident_Rate_Per_1000_Vehicles"]
X=df[features]
y=df["Danger_Level"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
pred=model.predict(X_test)
st.write("Accuracy:", round(accuracy_score(y_test,pred)*100,2),"%")
st.text("Confusion Matrix")
st.write(confusion_matrix(y_test,pred))
st.text("Classification Report")
st.text(classification_report(y_test,pred))

state_input=st.text_input("Enter State Name")

if st.button("Predict"):
    found=None
    for s in label_encoders["State"].classes_:
        if s.lower()==state_input.strip().lower():
            found=s
            break
    if found:
        enc=label_encoders["State"].transform([found])[0]
        data=df[df["State"]==enc]
        if len(data):
            sm=data.iloc[0]
            p=model.predict([[sm[f] for f in features]])
            res=label_encoders["Danger_Level"].inverse_transform(p)[0]
            st.subheader("Prediction Result")
            for f in features:
                st.write(f,":",sm[f])
            st.success(f"Predicted Danger Level: {res}")
            if res=="High":
                st.error("HIGH RISK")
                st.markdown("- Follow speed limits\n- Increase monitoring\n- Wear helmets and seat belts\n- Avoid drunk driving")
            elif res=="Medium":
                st.warning("MEDIUM RISK")
                st.markdown("- Follow traffic rules\n- Avoid overspeeding\n- Maintain vehicle")
            else:
                st.success("LOW RISK")
                st.markdown("- Continue safe driving\n- Maintain vehicle\n- Wear helmets and seat belts")
        else:
            st.error("No data available for this state.")
    else:
        st.error("State Name Not Found In Dataset")

imp=pd.DataFrame({"Feature":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
st.subheader("Feature Importance")
st.dataframe(imp)
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(imp["Feature"],imp["Importance"])
plt.xticks(rotation=90)
st.pyplot(fig)

st.subheader("State Wise Total Accidents")
st.write(df.groupby("State")["Total_Accidents"].sum())
st.success("Program Executed Successfully")
