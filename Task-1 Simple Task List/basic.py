import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

TASKS_FILE = "tasks.csv"

def load_tasks():
    if os.path.exists(TASKS_FILE):
        return pd.read_csv(TASKS_FILE)
    return pd.DataFrame(columns=["Task", "Priority"])

def save_tasks(df):
    df.to_csv(TASKS_FILE, index=False)

def add_task(task, priority):
    df = load_tasks()
    df = pd.concat([df, pd.DataFrame([[task, priority]], columns=["Task", "Priority"])], ignore_index=True)
    save_tasks(df)
    print("Task added successfully!")

def remove_task(task):
    df = load_tasks()
    df = df[df["Task"] != task]
    save_tasks(df)
    print("Task removed successfully!")

def list_tasks():
    df = load_tasks()
    if df.empty:
        print("No tasks available.")
    else:
        print(df)

def recommend_task():
    df = load_tasks()
    if df.empty:
        print("No tasks available for recommendations.")
        return
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Task"].values.astype('U'))
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    most_similar_task_idx = similarity_matrix.sum(axis=0).argmax()
    recommended_task = df.iloc[most_similar_task_idx]
    print(f"Recommended Task: {recommended_task['Task']} ({recommended_task['Priority']})")

def main():
    while True:
        print("\nTask Management App")
        print("1. Add Task")
        print("2. Remove Task")
        print("3. List Tasks")
        print("4. Recommend Task")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            task = input("Enter task description: ")
            priority = input("Enter task priority (High, Medium, Low): ")
            add_task(task, priority)
        elif choice == "2":
            task = input("Enter task description to remove: ")
            remove_task(task)
        elif choice == "3":
            list_tasks()
        elif choice == "4":
            recommend_task()
        elif choice == "5":
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
