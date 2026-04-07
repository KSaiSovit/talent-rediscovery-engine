import pandas as pd

data = {
    "Name": ["Alice Johnson", "Bob Smith", "Carla Diaz", "David Chen", "Emma Watson"],
    "Email": ["alice@example.com", "bob@example.com", "carla@example.com", "david@example.com", "emma@example.com"],
    "Phone": ["+1 555-0101", "+1 555-0102", "+1 555-0103", "+1 555-0104", "+1 555-0105"],
    "Current / last job title": ["Data Scientist", "Backend Engineer", "Product Manager", "ML Engineer", "UX Designer"],
    "Skills": [
        "Python, SQL, Machine Learning, TensorFlow, NLP",
        "Java, Spring Boot, AWS, Docker, PostgreSQL",
        "Product Strategy, Agile, JIRA, Figma, User Stories",
        "Python, PyTorch, NLP, Scikit-learn, Transformers",
        "Figma, Sketch, User Research, Prototyping, Accessibility"
    ],
    "Experience": ["5 years", "8 years", "6 years", "4 years", "7 years"],
    "Location": ["New York, NY", "Austin, TX", "San Francisco, CA", "Seattle, WA", "Chicago, IL"],
    "Current company": ["DataCorp", "TechSolve", "ProdWorks", "AI Labs", "DesignStudio"],
    "Notice period": ["2 weeks", "1 month", "Immediate", "2 weeks", "1 month"],
    "Salary / expected salary": ["$120k", "$135k", "$140k", "$115k", "$110k"],
    "Last interaction date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-01-05", "2023-02-28"],
    "Resume text or profile summary": [
        "Data scientist with 5 years experience in predictive modeling and deep learning. Built multiple NLP pipelines.",
        "Backend engineer skilled in microservices, cloud infrastructure, and database design.",
        "Product manager who led cross-functional teams to launch 3 B2B products.",
        "ML engineer focused on transformer models and productionising LLMs.",
        "User-centric designer with a passion for design systems and inclusive UX."
    ]
}
df = pd.DataFrame(data)
df.to_csv("sample_candidates.csv", index=False)
print("sample_candidates.csv created successfully.")