from notion_client import Client

NOTION_TOKEN = "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"
DATABASE_ID = "33682958b1f4803e9dc1cb0066f28195"

notion = Client(auth=NOTION_TOKEN)
response = notion.databases.query(database_id=DATABASE_ID)

for page in response["results"]:
    print("字段列表：")
    for key, value in page["properties"].items():
        print(f"  {key}: {value['type']} -> {value}")
