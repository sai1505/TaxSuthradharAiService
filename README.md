# 🧾 TaxSuthradhar

**TaxSuthradhar** is a **Tax Compliance Assistance System** designed to help IT employees **legally save money on taxes** by analyzing Income Tax Return (ITR) files. It provides actionable insights, historical data visualization, and a smooth user experience to simplify tax compliance.

**Note:** The User Interface service powered by FastAPI is hosted in a separate repository: [TaxSuthradhar](https://github.com/sai1505/TaxSuthradhar)

---

## 💡 Overview

Managing taxes can be complicated, especially for IT professionals with multiple deductions and exemptions. TaxSuthradhar simplifies this process by:  

- Analyzing ITR files for potential savings.  
- Providing interactive charts to visualize your tax history.  
- Allowing secure document uploads and management.  
- Offering a smart chat interface for guidance and queries.  

This system ensures that **you stay tax-compliant while maximizing legal savings**.

---

## 🌟 Features

### 🔐 Authentication
- Sign up and log in via **Google** or **Email & Password**.  
- Secure user profile management with username, display name, and email.

### 💬 Chat Interface
- Interactive chat for **tax guidance**.  
- Ask questions about exemptions, deductions, and compliance.

### 📊 History & Charts
- Track your **tax history over time**.  
- Interactive visualizations for better understanding of tax patterns.

### 📂 Document Management
- Upload and manage **ITR documents**.  
- Easily view which documents have been uploaded.  

### 👤 Profile
- Manage and view account details.  
- Display **username, display name, and email**.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React.js |
| Backend | Node.js, Express.js, FastAPI |
| AI / Automation | Langchain, GroQ Models, Docling |
| Storage / Database | Firebase, Cloudflare R2 |

---

## 📁 Folder Structure

```

TaxSuthradhar/
├─ frontend/         # React.js client
├─ backend/          # Node.js + Express + FastAPI
└─ README.md

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+ recommended)  
- npm or yarn  
- Python (for FastAPI)  
- Firebase account  
- Cloudflare R2 account (optional for storage)  

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sai1505/TaxSuthradhar.git
   cd TaxSuthradhar
````

2. Install frontend dependencies:

   ```bash
   cd client
   npm install
   ```

3. Install backend dependencies:

   ```bash
   cd ../server
   npm install
   ```

4. Configure environment variables:

   * Create a `.env` file in `server/` and `client/` directories using `.env.example` as a template.
   * Add Firebase API keys, Cloudflare R2 credentials, and other necessary configs.

5. Run the backend server:

   ```bash
   npm start  # For Node.js/Express
   uvicorn main:app --reload  # For FastAPI
   ```

6. Run the frontend:

   ```bash
   cd client
   npm start
   ```

---

## 🖥️ Usage

1. **Register / Login** with Google or Email.
2. **Upload ITR documents** for analysis.
3. **Use the chat interface** to ask questions about taxes.
4. **View history and charts** to track your past tax data.
5. **Check your profile** for account information.

---

## 💡 How It Works

1. Users upload their **ITR documents** to the system.
2. Backend parses the files and extracts relevant **financial data**.
3. AI models (Langchain + GroQ) analyze deductions, exemptions, and compliance.
4. Data is stored securely in **Firebase** or **Cloudflare R2**.
5. Users can **interact through the chat interface**, view history charts, and manage documents.

---

## 🔐 Security

* OAuth 2.0 authentication for Google login.
* Encrypted password storage for email/password accounts.
* Secure file storage on **Cloudflare R2 / Firebase**.

---

## 📈 Benefits for IT Employees

* **Legal tax savings** by maximizing eligible exemptions.
* **Simplified tax management** in one platform.
* **Interactive visualization** of financial history.
* **Secure and private** document handling.

---

## ✨ Acknowledgements

* [React.js](https://reactjs.org/)
* [Node.js](https://nodejs.org/)
* [Express.js](https://expressjs.com/)
* [Langchain](https://www.langchain.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Firebase](https://firebase.google.com/)
* [Cloudflare R2](https://developers.cloudflare.com/r2/)

---

Made with ❤️ by **sai1505** for IT employees to **simplify tax compliance and maximize savings**.
Its in development and Beta Phase yet.
