## Colab Link

```
https://colab.research.google.com/drive/1VP1ntAjiW7NbmujSTV5-buYEB2At8stB?usp=sharing
```

## create env

python3 -m venv chatbot-env

## run by active env

source chatbot-env/bin/activate

## run train model chatbot

python chatbot.py

## run pre-train model chatbot

python pre_train_chatbot.py

## run test with api request

python app.py

## api

```
curl -X POST http://127.0.0.1:5000/chatbot -H "Content-Type: application/json" -d '{"message": "hi"}'
```
