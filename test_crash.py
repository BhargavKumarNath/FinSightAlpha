import requests
query = "Identify the current CEO of the company that acquired Figma in 2022, and then list the primary programming language used in the open-source web framework that this CEO's company originally created."
try:
    res = requests.post('http://localhost:8000/chat', json={'query': query})
    print(res.status_code)
    print(res.text)
except Exception as e:
    print(e)
