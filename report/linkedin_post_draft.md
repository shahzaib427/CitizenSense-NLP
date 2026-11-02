# LinkedIn post - DRAFT (rewrite in your own words, as the brief requires)

Today, I completed AIRI Team PITB - AI Internship Task 4, where I built an end-to-end Natural Language Processing project.

For this task, I developed CitizenSense, an NLP-based system that analyzes text feedback (English, Roman Urdu and mixed) and predicts category, sentiment, priority, keywords, summary and a suggested department.

The project covered the complete NLP workflow:
- Text data collection (synthetic, clearly labelled)
- Data cleaning and preprocessing
- Label creation
- TF-IDF (word + character n-gram) features
- Baseline model training and comparison (NB, Logistic Regression, SVM, Random Forest)
- Hugging Face transformer usage
- Evaluation metrics
- Error analysis
- Demo interface (Streamlit / FastAPI / CLI)
- Final documentation

A key lesson: my model scored ~99% on a template-based test split but only ~59% on 49 hand-written messages. That gap taught me how easily NLP metrics can look better than reality when data lacks diversity - and why error analysis matters.

Tools used: Python, Google Colab, Pandas, Scikit-learn, TF-IDF, Hugging Face Transformers, Streamlit, FastAPI.

#NLP #NaturalLanguageProcessing #MachineLearning #Transformers #HuggingFace #Python #AIRI #PITB #AIInternship #TextClassification
