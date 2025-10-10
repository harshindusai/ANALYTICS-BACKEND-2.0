import requests

response = requests.post("http://localhost:8000/query", 
                        json={"query": "Show total sales by category"})
result = response.json()

print("SQL:", result['sql_query'])
print("Description:", result['description'])
if result['graph_url']:
    print("Graph:", f"http://localhost:8000{result['graph_url']}")
