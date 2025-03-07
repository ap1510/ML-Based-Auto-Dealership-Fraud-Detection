# Auto Dealership Fraud Detection 🚗💰

## 📌 Project Overview
This project identifies fraudulent practices in auto dealerships by analyzing **negative customer reviews** using **Machine Learning & NLP**. The pipeline:

1️⃣ **Data Collection** → Scraped **DealerRater** reviews using BeautifulSoup.  
2️⃣ **Filtering** → Kept only **negative reviews** to detect key dealership issues.  
3️⃣ **Clustering** → Used **DBSCAN & t-SNE** to identify common complaint patterns.  
4️⃣ **Classification Models** → Built **TF-IDF + SVM** models to detect:
   - **Advertised Price Discrepancies**  
   - **APR Issues**  
   - **Refund Problems**  
   - **Title Registration Delays** 
 
5️⃣ **Evaluation** → Optimized using **GridSearchCV & Cross-validation**.  

---

## 📂 Repository Structure
```
📦 Auto-Dealership-Fraud-Detection
 ┣ 📜 Auto_Dealership_Analysis_Summary.pdf # Summary of the Analysis
 ┣ 📂 Data/                    # Sample datasets (cleaned & raw)
 ┣ 📂 models/saved model                   # Saved trained models (.sav files)
 ┣ 📂 src/                      # Python scripts for each ML model
 ┣ 📂 Tf-Idf Vectors            # Word Embeddings for the reviews
 ┣ 📂 Data Extraction           # Webscraping Script
 ┣ 📂 Results                   # Results (CSV)
 ┣ 📜 README.md                 # Project documentation
 ┗ 📜 LICENSE                   # Open-source license
```

---


---

## 🔍 Model Performance & Evaluation
### **Confusion Matrix & F1 Scores** 📊
- Advertised Prices Model: **F1 Score: 0.87**  
- APR Issues Model: **F1 Score: 0.85**  
- Refund Model: **F1 Score: 0.82**  
- Title Issues Model: **F1 Score: 0.88**  

🚀 **Visualizations Coming Soon!**

---

## 🏆 Results & Business Impact
✅ Identified **fraudulent dealerships** across multiple states  
✅ Helped **consumers avoid scams** by flagging high-risk dealers  
✅ Demonstrated a scalable approach for **real-world fraud detection**  

---

## 🤝 Contributing
Want to improve this project? Feel free to **fork & submit PRs!**

---

## 📩 Contact
👤 **Aditya Parashar**  
📧 Email: adityaparashar1150@gmail.com  
🔗 LinkedIn: [linkedin.com/in/adityaparashar149](https://linkedin.com/in/adityaparashar149)  
📂 GitHub: [github.com/yourgithub](https://github.com/yourgithub)  

🚀 **Let’s build AI-driven fraud detection together!**

