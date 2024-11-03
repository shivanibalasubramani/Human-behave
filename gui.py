import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from tkinter import messagebox
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('data-final.csv', delimiter='\t')  
data.dropna(inplace=True)
continents = pd.read_csv('continents2.csv')

data1 = data.copy()
data1['alpha-2'] = data1['country']
continents = data1.merge(continents, on=["alpha-2"], how='left')

def perform_clustering():
    try:
        data_selected = continents[['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10']]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_selected)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        kmeans = KMeans(n_clusters=5, random_state=0)
        labels = kmeans.fit_predict(pca_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title('K-means Clustering with PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def plot_personality_traits(traits, title):
    plt.figure(figsize=[15, 15])
    for n, f in enumerate(traits, start=1):
        plt.subplot(5, 2, n)
        sns.countplot(x=f, edgecolor="black", alpha=0.7, data=data)
        plt.title(f"{title}: {f}")
    plt.tight_layout()
    plt.show()


print("Merged Continents DataFrame:")
print(continents.head())  
print("Unique regions in merged data:", continents['region'].unique())

def plot_continental_personality_traits(trait_list, title):
   
    df = continents.groupby(["region"])[trait_list].mean()  
    df.columns = trait_list
    df = df.loc[['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'], :]
    df = df.T

    df["World"] = df.mean(axis=1)  
    df = df.sort_values(by="World", ascending=True)
    df["Min"] = df.min(axis=1)
    df["Max"] = df.max(axis=1)

    fig = plt.figure(figsize=(10, 5))
    ax0 = fig.add_subplot(111)

    background_color = "#fbfbfb"
    fig.patch.set_facecolor(background_color)
    ax0.set_facecolor(background_color)

    y_dummy = np.arange(1, len(df.index) + 1)
    ax0.hlines(y=y_dummy, xmin=df["Min"], xmax=df["Max"], color='grey', alpha=0.4, zorder=3)
    ax0.scatter(df['World'], y_dummy, color='red', label='World')
    ax0.scatter(df['Africa'], y_dummy, color='green', label='Africa')
    ax0.scatter(df['Americas'], y_dummy, color='blue', label='Americas')
    ax0.scatter(df['Asia'], y_dummy, color='orange', label='Asia')
    ax0.scatter(df['Europe'], y_dummy, color='skyblue', label='Europe')
    ax0.scatter(df['Oceania'], y_dummy, color='magenta', label='Oceania')

    y_label = list(df.index)
    y_label.insert(0, "")
    ax0.set_yticklabels(y_label)
    ax0.set_xticklabels([])
    ax0.tick_params(bottom=False)

    ax0.text(-100, len(y_dummy) / 2, title, fontsize=20, fontweight='bold', fontfamily='serif')
    ax0.legend(loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.1))

    for s in ["top", "right", "left", "bottom"]:
        ax0.spines[s].set_visible(False)

    plt.show()


def show_extroversion():
    plot_personality_traits(['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'], "Extroversion Q&As")

def show_neuroticism():
    plot_personality_traits(['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'], "Neuroticism Q&As")

def show_agreeableness():
    plot_personality_traits(['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'], "Agreeableness Q&As")

def show_conscientiousness():
    plot_personality_traits(['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'], "Conscientiousness Q&As")

def show_openness():
    plot_personality_traits(['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'], "Openness Q&As")

def show_continental_extroversion():
    plot_continental_personality_traits(['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'], 'Visualization of Extraversion across Countries')

def show_continental_neuroticism():
    plot_continental_personality_traits(['EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'], 'Visualization of Neuroticism across Countries')

def show_continental_agreeableness():
    plot_continental_personality_traits(['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'], 'Visualization of Agreeableness across Countries')

def show_continental_conscientiousness():
    plot_continental_personality_traits(['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'], 'Visualization of Conscientiousness across Countries')

def show_continental_openness():
    plot_continental_personality_traits(['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'], 'Visualization of Openness across Countries')

def predict_personality():
    questions = [
        'I am the life of the party (Yes/No)',
        'I get stressed out easily (Yes/No)',
        'I feel little concern for others (Yes/No)',
        'I am always prepared (Yes/No)',
        'I have a rich vocabulary (Yes/No)'
    ]
    
    # Initialize response storage
    responses = []

    def get_response():
        response = response_var.get()
        if response not in ['Yes', 'No']:
            messagebox.showerror("Invalid Input", "Please enter 'Yes' or 'No'")
            return

        responses.append(1 if response == 'Yes' else 0)
        if len(responses) < len(questions):
            question_label.config(text=questions[len(responses)])
            response_var.set("")
        else:
            calculate_and_display_results()
            response_window.destroy()

    def calculate_and_display_results():
        traits = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        counts = np.array(responses)  
        percentages = (counts / len(questions)) * 100


        plt.figure(figsize=(8, 8))
        plt.pie(percentages, labels=traits, autopct='%1.1f%%', startangle=140)
        plt.title('Predicted Personality Traits Percentage')
        plt.axis('equal')  
        plt.show()

    
        final_predicted_trait = traits[np.argmax(percentages)]
        final_label.config(text=f"Final Predicted Trait: {final_predicted_trait}")

    
    response_window = Toplevel(root)
    response_window.title("Predict Personality")

    response_var = StringVar()
    question_label = Label(response_window, text=questions[0], font=("Arial", 14))
    question_label.pack(pady=20)

    entry = Entry(response_window, textvariable=response_var, font=("Arial", 14))
    entry.pack(pady=20)
    entry.bind("<Return>", lambda event: get_response())

    next_button = Button(response_window, text="Next", command=get_response)
    next_button.pack(pady=10)

    final_label = Label(response_window, text="", font=("Arial", 14))
    final_label.pack(pady=20)

def create_gui():
    global root
    root = Tk()
    root.title("Personality Traits Analysis")

    frame = Frame(root)
    frame.pack(pady=20)

    buttons = [
        ("Extroversion", show_extroversion),
        ("Neuroticism", show_neuroticism),
        ("Agreeableness", show_agreeableness),
        ("Conscientiousness", show_conscientiousness),
        ("Openness", show_openness),
        ("Continental Extraversion", show_continental_extroversion),
        ("Continental Neuroticism", show_continental_neuroticism),
        ("Continental Agreeableness", show_continental_agreeableness),
        ("Continental Conscientiousness", show_continental_conscientiousness),
        ("Continental Openness", show_continental_openness),
        ("Predict Personality", predict_personality),
        ("Perform Clustering", perform_clustering)
    ]

    for (text, command) in buttons:
        Button(frame, text=text, command=command).pack(pady=5)

    root.mainloop()

create_gui()