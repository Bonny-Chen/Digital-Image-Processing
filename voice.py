import speech_recognition as sr

r = sr.Recognizer()
m = sr.Microphone()
with m as source:
    print("speak")
    audio= r.listen(source)
    try:
        text=r.recognize_google(audio)
        print("say{}".format(text))
    except:
        print("sorry")