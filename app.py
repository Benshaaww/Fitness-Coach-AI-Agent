#app.py refined by google gemini 3 pro
import tkinter as tk
from tkinter import scrolledtext
import threading
from datetime import datetime
from main import ChatBotAssistant

COLOR_BG = "#1E1E1E"
COLOR_SIDEBAR = "#252526"
COLOR_ACCENT = "#4CAF50"
COLOR_TEXT = "#E0E0E0"
COLOR_INPUT = "#333333"
COLOR_USER_MSG = "#2D3748"
COLOR_BOT_MSG = "#333333"

class FitnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Convy Fitness Coach")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLOR_BG)
        
        print("Loading AI Model...")
        self.bot = ChatBotAssistant('intents.json')
        self.bot.parse_intents()
        self.bot.prepare_data()
        self.bot.train_model(batch_size=8, lr=0.001, epochs=50)
        print("AI Ready.")

        self.sidebar_frame = tk.Frame(root, bg=COLOR_SIDEBAR, width=280)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        self.main_frame = tk.Frame(root, bg=COLOR_BG)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.title_label = tk.Label(self.sidebar_frame, text="FITNESS\nCOACH", font=("Helvetica", 24, "bold"), bg=COLOR_SIDEBAR, fg=COLOR_ACCENT, justify=tk.LEFT)
        self.title_label.pack(pady=(40, 20), padx=20, anchor="w")

        self.time_label = tk.Label(self.sidebar_frame, text="00:00", font=("Helvetica", 36), bg=COLOR_SIDEBAR, fg=COLOR_TEXT)
        self.time_label.pack(pady=(10, 0), anchor="center")
        
        self.date_label = tk.Label(self.sidebar_frame, text="Monday", font=("Helvetica", 12), bg=COLOR_SIDEBAR, fg="#888888")
        self.date_label.pack(pady=(0, 30), anchor="center")
        self.update_time()

        self.coach_label = tk.Label(self.sidebar_frame, text="YOUR COACH", font=("Arial", 10, "bold"), bg=COLOR_SIDEBAR, fg="#666666")
        self.coach_label.pack(pady=(0, 10))

        self.canvas = tk.Canvas(self.sidebar_frame, width=200, height=250, bg=COLOR_SIDEBAR, highlightthickness=0)
        self.canvas.pack()

        self.head = self.canvas.create_oval(75, 50, 125, 100, fill=COLOR_TEXT, width=0)
        self.body = self.canvas.create_rectangle(85, 100, 115, 180, fill=COLOR_ACCENT, width=0)
        self.arm_l = self.canvas.create_line(85, 110, 60, 140, width=8, fill=COLOR_TEXT, capstyle=tk.ROUND)
        self.arm_r = self.canvas.create_line(115, 110, 140, 140, width=8, fill=COLOR_TEXT, capstyle=tk.ROUND)
        self.leg_l = self.canvas.create_line(90, 180, 80, 230, width=8, fill="#555555", capstyle=tk.ROUND)
        self.leg_r = self.canvas.create_line(110, 180, 120, 230, width=8, fill="#555555", capstyle=tk.ROUND)

        self.is_god_mode = False
        self.anim_offset = 0
        self.anim_direction = 1
        self.animate_coach()

        self.chat_area = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, font=("Helvetica", 11), bg=COLOR_BG, fg=COLOR_TEXT, insertbackground=COLOR_ACCENT, relief=tk.FLAT, padx=20, pady=20)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.chat_area.config(state=tk.DISABLED)

        self.chat_area.tag_config('user', background=COLOR_USER_MSG, foreground=COLOR_TEXT, rmargin=50, lmargin1=100, lmargin2=100, justify='right', spacing1=10, spacing3=10)
        self.chat_area.tag_config('bot', background=COLOR_BOT_MSG, foreground=COLOR_ACCENT, rmargin=100, lmargin1=50, lmargin2=50, justify='left', spacing1=10, spacing3=10)
        self.chat_area.tag_config('small', font=("Arial", 8), foreground="#666666")

        self.input_container = tk.Frame(self.main_frame, bg=COLOR_BG)
        self.input_container.pack(fill=tk.X, padx=20, pady=(0, 30))

        self.entry_field = tk.Entry(self.input_container, font=("Helvetica", 14), bg=COLOR_INPUT, fg=COLOR_TEXT, insertbackground=COLOR_ACCENT, relief=tk.FLAT)
        self.entry_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 15))
        self.entry_field.bind("<Return>", lambda event: self.send_message())

        self.send_btn = tk.Button(self.input_container, text="SEND", command=self.send_message, bg=COLOR_ACCENT, fg="#000000", font=("Arial", 10, "bold"), relief=tk.FLAT, padx=20, activebackground="#388E3C")
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)

        self.display_message("Bot", "Welcome back. Ready to train?")

    def update_time(self):
        now = datetime.now()
        self.time_label.config(text=now.strftime("%H:%M"))
        self.date_label.config(text=now.strftime("%A, %d %B"))
        self.root.after(1000, self.update_time)

    def animate_coach(self):
        try:
            self.canvas.delete("eyes") 
            self.canvas.delete("mouth")
            
            if self.is_god_mode:
                bounce = 8; speed = 40
                

                eye_size = 3 + abs(self.anim_offset / 1.5) 
                

                self.canvas.create_oval(90-eye_size, 75-eye_size, 90+eye_size, 75+eye_size, fill="red", tags="eyes")
                self.canvas.create_line(85, 70, 95, 78, width=2, fill="black", tags="eyes")
                

                self.canvas.create_oval(110-eye_size, 75-eye_size, 110+eye_size, 75+eye_size, fill="red", tags="eyes")
                self.canvas.create_line(105, 78, 115, 70, width=2, fill="black", tags="eyes")


                self.canvas.create_arc(90, 85, 110, 105, start=0, extent=180, style=tk.ARC, width=3, outline="black", tags="mouth")

            else:
                bounce = 2; speed = 600
                

                self.canvas.create_oval(92, 72, 98, 78, fill="black", tags="eyes")
                self.canvas.create_oval(102, 72, 108, 78, fill="black", tags="eyes")


                self.canvas.create_arc(90, 80, 110, 100, start=180, extent=180, style=tk.ARC, width=3, outline="black", tags="mouth")
            
            self.anim_offset += self.anim_direction * bounce
            if abs(self.anim_offset) > 8: self.anim_direction *= -1

            dy = self.anim_direction * (bounce * 0.5)
            
            self.canvas.move(self.head, 0, dy)
            self.canvas.move(self.body, 0, dy)
            self.canvas.move(self.arm_l, 0, dy)
            self.canvas.move(self.arm_r, 0, dy)
            self.canvas.move(self.leg_l, 0, dy)
            self.canvas.move(self.leg_r, 0, dy)
            self.canvas.move("eyes", 0, dy) 
            self.canvas.move("mouth", 0, dy)

            self.root.after(speed, self.animate_coach)
        except Exception as e:
            print(f"Animation Error: {e}", flush=True)

    def send_message(self):
        msg = self.entry_field.get()
        if not msg.strip(): return
        self.entry_field.delete(0, tk.END)
        self.display_message("You", msg)
        
        if msg.lower() in ['exit', 'quit']:
            self.root.quit()
        else:
            threading.Thread(target=self.get_response, args=(msg,)).start()

    def get_response(self, msg):
        response, intent = self.bot.process_message(msg)
        if intent == 'secret_mode': self.is_god_mode = True
        elif intent == 'greeting': self.is_god_mode = False
        self.root.after(0, lambda: self.display_message("Bot", response))
        if intent == 'goodbye': self.root.after(2000, self.root.quit)

    def display_message(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        if sender == "You":
            self.chat_area.insert(tk.END, f"\n{message}\n", 'user')
        else:
            self.chat_area.insert(tk.END, f"\n{message}\n", 'bot')
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

if __name__ == "__main__":
    try:
        print("Initializing App...", flush=True)
        root = tk.Tk()
        app = FitnessApp(root)
        print("Starting Mainloop...", flush=True)
        root.mainloop()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        input("Press Enter to exit...")
