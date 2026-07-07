print("=" * 50)
print("🤖 SMART RULE-BASED CHATBOT")
print("Type 'bye' to exit")
print("=" * 50)

while True:
    user = input("\nYou: ").lower()
    # Greetings
    if user in ["hi", "hello", "hey", "hii", "heyy"]:
        print("Bot: Hello! Nice to meet you.")

    elif "good morning" in user:
        print("Bot: Good Morning! Have a productive day.")

    elif "good afternoon" in user:
        print("Bot: Good Afternoon!")

    elif "good evening" in user:
        print("Bot: Good Evening!")

    # Basic Questions
    elif "how are you" in user:
        print("Bot: I am doing great. Thanks for asking!")

    elif "your name" in user:
        print("Bot: My name is SmartBot.")

    elif "who created you" in user:
        print("Bot: I was created using Python programming.")

    elif "your age" in user:
        print("Bot: I don't have an age because I am a chatbot.")

    # User Information
    elif "my name is" in user:
        name = user.replace("my name is", "")
        print(f"Bot: Nice to meet you {name.title()}!")

    elif "i am" in user:
        print("Bot: That's nice to know!")

    # Education
    elif "college" in user:
        print("Bot: College life is a great opportunity to learn.")

    elif "study" in user:
        print("Bot: Consistent study leads to success.")

    elif "exam" in user:
        print("Bot: Start preparing early and revise regularly.")

    elif "assignment" in user:
        print("Bot: Complete assignments before the deadline.")

    elif "project" in user:
        print("Bot: Projects improve practical knowledge.")

    # Programming
    elif "python" in user:
        print("Bot: Python is easy to learn and widely used.")

    elif "java" in user:
        print("Bot: Java is an object-oriented programming language.")

    elif "html" in user:
        print("Bot: HTML is used to create web pages.")

    elif "css" in user:
        print("Bot: CSS is used to style web pages.")

    elif "javascript" in user:
        print("Bot: JavaScript adds interactivity to websites.")

    elif "dbms" in user:
        print("Bot: DBMS stands for Database Management System.")

    elif "sql" in user:
        print("Bot: SQL is used to manage databases.")

    elif "dsa" in user:
        print("Bot: DSA stands for Data Structures and Algorithms.")

    elif "ai" in user:
        print("Bot: AI enables machines to mimic human intelligence.")

    elif "machine learning" in user:
        print("Bot: Machine Learning is a branch of AI.")

    # Career
    elif "internship" in user:
        print("Bot: Internships help gain industry experience.")

    elif "job" in user:
        print("Bot: Keep improving your skills and build projects.")

    elif "resume" in user:
        print("Bot: A good resume highlights your skills and achievements.")

    elif "linkedin" in user:
        print("Bot: LinkedIn helps build your professional network.")

    # Motivation
    elif "motivate me" in user:
        print("Bot: Success is the result of daily efforts.")

    elif "sad" in user:
        print("Bot: Tough times don't last, but tough people do.")

    elif "failure" in user:
        print("Bot: Failure is a stepping stone to success.")

    elif "success" in user:
        print("Bot: Success comes from consistency and hard work.")

    # Fun
    elif "joke" in user:
        print("Bot: Why do programmers hate nature? Too many bugs!")

    elif "fun fact" in user:
        print("Bot: The first computer bug was an actual bug.")

    elif "favorite color" in user:
        print("Bot: I like all colors equally.")

    elif "favorite food" in user:
        print("Bot: I don't eat food, but pizza sounds tasty.")

    elif "movie" in user:
        print("Bot: I enjoy talking about technology more than movies!")

    # Health
    elif "gym" in user:
        print("Bot: Regular exercise keeps you healthy.")

    elif "health" in user:
        print("Bot: Drink water, sleep well, and exercise regularly.")

    elif "weight loss" in user:
        print("Bot: A balanced diet and exercise help in weight loss.")

    # Date and Time
    elif "time" in user:
        from datetime import datetime
        print("Bot: Current Time:", datetime.now().strftime("%H:%M:%S"))

    elif "date" in user:
        from datetime import datetime
        print("Bot: Today's Date:", datetime.now().strftime("%d-%m-%Y"))

    elif "day" in user:
        from datetime import datetime
        print("Bot:", datetime.now().strftime("%A"))

    # Mathematics
    elif "2+2" in user:
        print("Bot: 4")

    elif "5*5" in user:
        print("Bot: 25")

    elif "10/2" in user:
        print("Bot: 5")

    elif "100-50" in user:
        print("Bot: 50")

    # Help Section
    elif "help" in user:
        print("""
Bot: I can answer questions about:
✓ Greetings
✓ Programming
✓ Education
✓ Career
✓ Health
✓ Motivation
✓ Date & Time
✓ Jokes
✓ Basic Math
        """)

    # Thanks
    elif "thanks" in user or "thank you" in user:
        print("Bot: You're welcome!")

    # Goodbye
    elif user in ["bye", "goodbye", "exit", "quit"]:
        print("Bot: Goodbye! Have a wonderful day.")
        break

    # Default Response
    else:
        print("Bot: Sorry, I didn't understand that. Type 'help' to see available commands.")
