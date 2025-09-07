import streamlit as st
import pandas as pd
import numpy as np
import time,random
import os,certifi 
import sendgrid 
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import ZFeatureMap, TwoLocal
from qiskit.primitives import Sampler
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

st.set_page_config(page_title="IBM Quantum Jobs Tracker", layout="wide")
st.title("IBM Quantum Jobs Tracker")
token = st.text_input("Enter IBM Quantum API Token:", type="password")

if st.button("Fetch Jobs") and token:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    backends = ['ibmq_qasm_simulator', 'ibmq_lima', 'ibmq_belem']
    statuses = ['RUNNING', 'COMPLETE', 'FAIL']

    rows = []
    for i in range(20):
        rows.append({
             'Job ID': f'JOB{i+1}',
             'Backend': random.choice(backends),
             'Status': random.choice(statuses),
             'Creation Time': '2025-08-25T12:00:00',
             'Run Time (s)': random.randint(10, 200)
          })

    df = pd.DataFrame(rows)
    st.subheader("Job Overview")

    col_left, col_right = st.columns([1, 1])  
    with col_left:
        st.markdown("Job Summary")

# Count jobs
        total_jobs = len(df)
        running_count = (df["Status"] == "RUNNING").sum()
        completed_count = (df["Status"] == "COMPLETE").sum()
        failed_count = (df["Status"] == "FAIL").sum()

#2x2 grid for metrics 
        mcol1, mcol2 = st.columns(2)
        mcol3, mcol4 = st.columns(2)

        with mcol1:
            st.metric("Total Jobs", total_jobs)
        with mcol2:
            st.metric("Running", running_count)
        with mcol3:
            st.metric("Completed", completed_count)
        with mcol4:
            st.metric("Failed", failed_count)

    with col_right:
        st.markdown("Jobs by Status")
        status_counts = df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        fig_pie = px.pie(
            status_counts,
            names="Status",
            values="Count",
            color="Status",
            hole=0.3
         )
        st.plotly_chart(fig_pie, use_container_width=True)

    if df.shape[0] > 1:
        st.subheader("Job Status Prediction using Quantum ML")

 # Encode categorical data
        df['Backend_encoded'] = LabelEncoder().fit_transform(df['Backend'])
        df['Status_encoded'] = df['Status'].map(lambda x: 1 if x == "COMPLETE" else 0)

 # Scale features
        X = StandardScaler().fit_transform(df[['Backend_encoded']])
        y = df['Status_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   
        num_qubits = X_train.shape[1]  

        feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)
        ansatz = TwoLocal(num_qubits=num_qubits, rotation_blocks='ry', entanglement_blocks='cz', reps=1)

        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            sampler=Sampler()  #  Qiskit ML API
            )
        y_train_array = y_train.to_numpy().astype(int)
        vqc.fit(X_train, y_train_array)
        y_pred = vqc.predict(X_test)
        predicted_labels = ["COMPLETE" if i == 1 else "FAIL" for i in y_pred]

# DataFrame 
        results_df = pd.DataFrame({
             "Job ID": df.iloc[y_test.index]["Job ID"].values,
             "Backend": df.iloc[y_test.index]["Backend"].values,
             "Creation Time": df.iloc[y_test.index]["Creation Time"].values,
             "Run Time (s)": df.iloc[y_test.index]["Run Time (s)"].values,
             "Actual Status": df.iloc[y_test.index]["Status"].values,
             "Predicted Status": predicted_labels
                 })
        st.subheader("Prediction Results")
        st.dataframe(results_df, use_container_width=True)
#backend plot
        if not df.empty and "Backend" in df.columns: 
            st.subheader("Jobs by Backend")
            backend_counts = df['Backend'].value_counts().reset_index()
            backend_counts.columns = ['Backend', 'Job Count']
            fig_backend = px.bar(backend_counts, x='Job Count', y='Backend', color='Backend', text='Job Count',orientation='h')
            st.plotly_chart(fig_backend, use_container_width=True)

        if "jobs" not in st.session_state:
            st.session_state.jobs = []       
        if "last_add" not in st.session_state:   # 👈 this is new
            st.session_state.last_add =0   

# generate a fake job
        if time.time() - st.session_state.last_add > 5:
            new_job = {
                   "Job ID": f"JOB_{len(st.session_state.jobs)+1}",
                   "Backend": random.choice(["ibmq_qasm_simulator", "ibmq_lima", "ibmq_belem"]),
                   "Creation Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "Run Time (s)": random.randint(1, 50),
                   "Status": random.choice(["RUNNING", "COMPLETE", "FAIL"])
                  }
            st.session_state.jobs.append(new_job)
            st.session_state.last_add = time.time()
 # For mail           
            os.environ['SSL_CERT_FILE'] = certifi.where()
            SENDGRID_API_KEY = "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"
            SENDER_EMAIL = "pavitramalla000@gmail.com"
            RECEIVER_EMAIL = "pavitramalla000@gmail.com"

            email_body = f"""Hello {RECEIVER_EMAIL},
                                 A new quantum job has just arrived on the IBM Quantum Platform!
                                 
                                🔹 Job ID: {new_job['Job ID']}  
                                🔹 Backend: {new_job['Backend']}  
                                🔹 Status: {new_job['Status']}  
                                🔹 Duration: {new_job['Run Time (s)']} seconds  
                                🔹 Created At: {new_job['Creation Time']}
                                
                                Check your Job Tracker Dashboard for more details.
                                – Team Quantum"""
            message = Mail(from_email=SENDER_EMAIL,to_emails=RECEIVER_EMAIL,
                   subject=f"New Quantum Job Arrived: {new_job['Job ID']}",
                   plain_text_content=email_body)

            sg = SendGridAPIClient(SENDGRID_API_KEY)
            sg.send(message)
            st.success(f" Email sent for new job: {new_job['Job ID']}")
            df = pd.DataFrame(st.session_state.jobs)
            st.dataframe(df, use_container_width=True)                  

        

