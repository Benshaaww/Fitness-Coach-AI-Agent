import os # needed for file paths
import json # for reading the intents file
import random # used to pick random responses
import nltk # natural language toolkit - super important!
import numpy as np 
import torch # pytorch for the neural net
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import re # regex for extracting numbers


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ChatBotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatBotModel, self).__init__() # initialize the parent class


        self.fc1 = nn.Linear(input_size, 128) #neurons - maybe add more later?
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, output_size)
        #activation function
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x # return the output

        

class ChatBotAssistant:

    def __init__(self, intents_path):
        self.model = None # start with no model
        self.intents_path = intents_path


        self.documents = []
        self.vocabulary = [] #docs and vocab will be used to turn our sentances into numbers 
        self.intents = [] #Need in order to assign probabilities 
        self.intents_responses = {} #List we can choose from, from the output

        self.X = None
        self.y = None 
        
        # State interactions
        self.user_profile = {} # Context memory: stores weight, height, goal, etc.
        self.function_mappings = {
            'greeting': self.handle_greeting,
            'workout_plan': self.handle_workout_plan,
            'weight_loss': self.handle_advice,
            'muscle_gain': self.handle_advice,
            'calories': self.handle_calories,
            'bmi': self.handle_bmi,
            'save_weight': self.handle_save_weight,
            'save_height': self.handle_save_height,
            'motivation': self.handle_motivation,
            'secret_mode': self.handle_secret_mode
        }

    # fun easter egg (: 
    def handle_secret_mode(self, user_input):
        return """
      (•_•)
      <)   )╯  I AM THE APEX PREDATOR!
      /    \\
      
    No pain, no gain. Now drop and give me 20! 
    """

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words] # lower case everything and get the root word

        return words
        
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self): 
        lemmatizer = nltk.WordNetLemmatizer()
        
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
        
            self.intents = []
            self.intents_responses = {}
            self.vocabulary = []
            self.documents = []
            
            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))
                self.intents = sorted(set(self.intents))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss() # good for classification
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # adam is the best optimizer apparently


        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward() # backpropagation magic
                optimizer.step() # update weights
                running_loss += loss


            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss: {running_loss / len(dataloader):.4f}")
        
    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        
        self.model = ChatBotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    # --- DYNAMIC HANDLERS ---
    
    def handle_greeting(self, user_input):
        return None  # Return None to use default static responses

    def handle_advice(self, user_input):
        return None # Return None to use default static responses

    def handle_workout_plan(self, user_input):
        # Could add logic here to customize based on user_profile
        return None

    def handle_motivation(self, user_input):
        return None

    def handle_save_weight(self, user_input):
        # Simple extraction using regex for digits
        matches = re.findall(r'(\d+)', user_input)
        if matches:
            weight = int(matches[0])
            
            # Check for Pounds vs KG confusion
            if weight > 150:
                return f"Wow, {weight} seems high for kg! Are you using pounds? Please tell me your weight in kg (e.g. 'My weight is 80kg')."
            
            self.user_profile['weight'] = weight # save to dictionary

            return f"Great! I've saved your weight as {weight}kg."
        return "I couldn't catch that weight. Please say it like 'I weigh 80'."

    def handle_save_height(self, user_input):
        matches = re.findall(r'(\d+)', user_input)
        if matches:
            height = int(matches[0])
            self.user_profile['height'] = height
            return f"Got it. Your height is {height}cm."
        return "I didn't quite get that. Please say something like 'My height is 180'."

    def handle_calories(self, user_input):
        if 'weight' not in self.user_profile:
            return "I need to know your weight to calculate calories. Please tell me 'My weight is X kg'."
        
        weight = self.user_profile['weight']
        maintenance = weight * 24 # crude formula
        
        return f"Based on your weight of {weight}kg, your estimated maintenance calories are approx {maintenance} per day. To lose weight, aim for {maintenance - 500}."

    def handle_bmi(self, user_input):
        if 'weight' not in self.user_profile or 'height' not in self.user_profile:
             return "I need both your weight (in kg) and height (in cm) to calculate BMI. Please tell me them separately."
        
        w = self.user_profile['weight']
        # Check if height is reasonable for cm
        h = self.user_profile['height']
        if h < 3.0: # assume meters if small
             h = h * 100
             
        h_m = h / 100.0 # convert cm to m
        
        bmi = w / (h_m * h_m)
        category = "Natural"
        if bmi < 18.5: category = "Underweight"
        elif bmi < 25: category = "Normal weight"
        elif bmi < 30: category = "Overweight"
        else: category = "Obese"
        
        return f"Your BMI is {bmi:.1f}, which is considered {category}."


    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor(bag, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            
        probabilities = torch.softmax(predictions, dim=0) # turn raw scores into probabilities

        predicted_class_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_index].item()
        
        predicted_tag = self.intents[predicted_class_index]
        
        print(f"DEBUG: Predicted: {predicted_tag} ({confidence:.2f})")

        if confidence < 0.35: # Lowered threshold
             return "I'm not sure I understand. Can you rephrase that? (Try asking about workouts, diet, or weight)", None

        # Check for dynamic handler
        response = None
        if predicted_tag in self.function_mappings:
            response = self.function_mappings[predicted_tag](input_message)
            
        # If dynamic handler returned a string, use it. Otherwise random choice from static responses.
        if response:
            return response, predicted_tag
        
        if predicted_tag in self.intents_responses:
            return random.choice(self.intents_responses[predicted_tag]), predicted_tag
            
        return "I'm not sure how to respond to that.", None

if __name__ == '__main__':
    # Initialize and train
    assistant = ChatBotAssistant('intents.json')
    assistant.parse_intents()
    assistant.prepare_data()
    
    print("Training model on new fitness intents...")
    assistant.train_model(batch_size=8, lr=0.001, epochs=150)
    assistant.save_model('chatbot_model.pth', 'dimensions.json')
    print("Training complete! AI Fitness Coach is ready.")
    
    print("\n--- AI FITNESS COACH STARTED ---")
    print("Try saying: 'Hi', 'I want to lose weight', 'My weight is 80kg', 'Calculate calories'")
    
    while True:
        try:
            message = input('You: ')
            if message.lower() in ['/quit', 'exit']:
                break
            
            response, intent = assistant.process_message(message)
            print(f"Bot: {response}")
            
            if intent == 'goodbye':
                break
        except KeyboardInterrupt:
            break