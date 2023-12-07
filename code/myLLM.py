from secret.secret import apiKey
from langchain.llms import GooglePalm

chat = GooglePalm(google_api_key=apiKey,temperature=0.8)

response = chat("""Act as a PPT creator for fees structure for IITM Bs Degree students.
                create a professional powerpoint presentation please show the complete fees in different slides
                for all categories because the main aim is to show the fees of IITM""")


print(response)