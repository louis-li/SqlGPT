<h1 align="center">
  ChatGPT with your data powered by Azure OpenAI  💡
</h1>

<p align="center">
  <strong>Documentation Assistant with Azure OpenAI</strong>
</p>

<p align="left">
  <strong>PdfGPT with Azure OpenAI</strong> is a sample Question and Answering bot using Azure OpenAI. It's designed to demonstrate how to use your own data for QnA as well as how to use Index to answer questions.
  
</p>


## Project structure
- Application - flask app (main application) with a simple HTML as frontend

## QuickStart

Note: Make sure you have docker installed

1. Open dowload this repository with `git clone https://github.com/louis-li/PdfGPT.git`
2. in application folder, mv .env.sample .env
3. Edit .env file and add your Azure OpenAI key and end point
3. Run `docker-compose build && docker-compose up`
4. Navigate to http://localhost:5010/
5. Use Upload button to upload PDF files and name it with your desired Index Name (stored in Redis)

To stop just run Ctrl + C



Built with [🦜️🔗 LangChain](https://github.com/hwchase17/langchain)

