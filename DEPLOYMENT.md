# Stock Market Analysis Project Deployment Guide

This guide explains how to deploy the Stock Market Analysis project using Streamlit.

## Local Deployment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit App**
   ```bash
   streamlit run src/app.py
   ```

3. **Access the Dashboard**
   - Open your web browser
   - Go to http://localhost:8501

## Cloud Deployment Options

### 1. Streamlit Cloud (Recommended)

1. **Create a GitHub Repository**
   - Push your code to GitHub
   - Make sure to include:
     - `requirements.txt`
     - `src/` directory
     - `README.md`

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `src/app.py`
   - Click "Deploy"

### 2. Heroku Deployment

1. **Create a Procfile**
   ```
   web: streamlit run src/app.py
   ```

2. **Create runtime.txt**
   ```
   python-3.9.16
   ```

3. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### 3. AWS EC2 Deployment

1. **Launch an EC2 Instance**
   - Choose Ubuntu Server
   - Configure security group to allow port 8501

2. **Install Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Run the App**
   ```bash
   streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
   ```

## Environment Variables

The following environment variables can be configured:

- `STREAMLIT_SERVER_PORT`: Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

## Security Considerations

1. **API Keys**
   - Keep any API keys in environment variables
   - Never commit sensitive data to the repository

2. **Access Control**
   - Consider implementing authentication for production deployments
   - Use HTTPS for secure data transmission

## Monitoring and Maintenance

1. **Logs**
   - Monitor application logs for errors
   - Set up log rotation

2. **Updates**
   - Regularly update dependencies
   - Monitor for security patches

## Troubleshooting

Common issues and solutions:

1. **Port Already in Use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   # Kill the process
   kill -9 <PID>
   ```

2. **Memory Issues**
   - Increase swap space
   - Optimize data processing

3. **Connection Issues**
   - Check firewall settings
   - Verify network configuration

## Support

For issues and support:
1. Check the project documentation
2. Open an issue on GitHub
3. Contact the development team 