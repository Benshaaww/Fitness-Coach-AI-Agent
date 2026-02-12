import os
from main import ChatBotAssistant # importing my cool bot
import sys # system stuff


print("--- Starting Verification ---")

try:
    # 1. Initialize
    bot = ChatBotAssistant('intents.json')
    bot.parse_intents()
    bot.prepare_data()

    # 2. Train (fast)
    print("Training...")
    # Faster training for quick verification
    bot.train_model(batch_size=8, lr=0.01, epochs=50) # gotta go fast!
 

    # 3. Test Intent & State
    print("Testing Logic...")

    # Test 1: Set Weight
    # This also tests if regex extraction is working
    test_msg = "My weight is 80kg"
    print(f"Testing Message: '{test_msg}'")
    resp, _ = bot.process_message(test_msg)
    print(f"Response: {resp}")
    
    if bot.user_profile.get('weight') == 80:
        print("PASS: Weight saved to profile correctly. Yesss!")
    else:
        print(f"FAIL: Weight not saved. Profile state: {bot.user_profile} :(")


    # Test 2: Calculate Calories (Dynamic)
    # This tests if the state from Test 1 is used
    if bot.user_profile.get('weight') == 80:
        test_msg = "Calculate calories"
        print(f"Testing Message: '{test_msg}'")
        resp, _ = bot.process_message(test_msg)
        print(f"Response: {resp}")
        
        # Maintenance for 80kg is approx 1920
        if "1920" in resp:
             print("PASS: Calorie calculation is correct. Math is hard but computers are smart.")

        else:
             print("FAIL: Calorie calculation incorrect.")
    else:
        print("SKIP: Calorie test skipped because weight wasn't saved.")

    # Test 3: Secret Mode (Easter Egg)
    print("Testing Secret Mode...")
    resp, _ = bot.process_message("Activate god mode")
    try:
        print(f"Secret Resp: {resp}")
    except:
        print("Secret Resp: [Cannot print characters]")
    
    if "APEX PREDATOR" in resp:
        print("PASS: Easter egg activated! (hope nobody finds this too early)")
    else:
        print("FAIL: Easter egg not found. Did I break the secret code?")


    # Test 4: Unit Check (High Weight)
    print("Testing Unit Check...")
    resp, _ = bot.process_message("My weight is 200")
    print(f"High Weight Resp: {resp}")
    if "seems high" in resp:
        print("PASS: Caught high weight correctly. No 200kg humans here.")

    else:
        print("FAIL: Did not catch high weight.")

    # Test 5: "Everything" Intent
    print("Testing 'Everything' Intent...")
    resp, tag = bot.process_message("Everything")
    print(f"Everything Resp: {resp} (Tag: {tag})")
    if tag == 'workout_options_all':
        print("PASS: Correctly identified 'everything' intent.")
    else:
        print(f"FAIL: Expected 'workout_options_all', got '{tag}'")

    # Test 6: "Stay Healthy" Intent
    print("Testing 'Stay Healthy' Intent...")
    resp, tag = bot.process_message("Stay healthy")
    print(f"Healthy Resp: {resp} (Tag: {tag})")
    if tag == 'general_health':
        print("PASS: Correctly identified 'general_health' intent.")
    else:
        print(f"FAIL: Expected 'general_health', got '{tag}'")

    print("\n--- Verification Finished ---")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
