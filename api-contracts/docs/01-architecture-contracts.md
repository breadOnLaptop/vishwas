# 📘 Design Document: The API Contract Layer

## 1. Overview & Purpose

In the **Vishwas** Hybrid-RAG architecture, three different programming languages must cooperate seamlessly:

* **Go (Orchestrator):** Handles high-concurrency traffic.
* **Python (Intelligence):** Runs AI/ML models.
* **C#/.NET (Knowledge):** Manages vector data and structure.

The **Contract Layer** is the single source of truth for communication between these services. Instead of guessing JSON field names (which leads to runtime errors), we define strict **Protocol Buffer (Protobuf)** schemas. This ensures that if the Python service changes its output format, the Go service will fail to compile immediately, preventing bugs from reaching production.

## 2. Design Needs & Choices

### 2.1 Why gRPC & Protobuf?

We selected **gRPC** over REST for internal communication for three specific reasons:

1. **Strict Typing:** REST APIs are "stringly typed" (everything is text). gRPC enforces data types (e.g., ensuring an Embedding is always a `list of floats`, not a string).
2. **Performance:** gRPC uses **HTTP/2** and binary serialization, making it significantly faster for sending large payloads (like PDF content or vector arrays) than JSON.
3. **Code Generation:** We define the API *once* in a `.proto` file, and the compiler generates the Go, Python, and C# client code automatically.

### 2.2 The "Handshake" Workflow

The system relies on two primary internal handshakes:

1. **The Extraction Handshake (Go ↔ Python):** Go hands over a raw file; Python returns structured "Chunks" and "Vectors".
2. **The Storage Handshake (Go ↔ .NET):** Go hands over the structured data; .NET confirms secure storage in PostgreSQL.

---

## 3. Service Specifications

### 3.1 Intelligence Service (`intelligence.proto`)

* **Owner:** Python Service
* **Role:** The "Brain" – Stateless processing unit.
* **Design Constraint:** Must handle large binary files (PDFs) and return high-dimensional vector arrays.

#### **RPC Methods**

| Method | Input | Output | Purpose |
| --- | --- | --- | --- |
| `ProcessDocument` | `DocumentRequest` (Raw Bytes) | `DocumentResponse` (Chunks + Vectors) | Takes a file, extracts text, chunks it, and generates embeddings. |
| `EmbedQuery` | `QueryRequest` (String) | `QueryVector` (Float Array) | Converts a user's question (e.g., "Is this fake?") into a vector for searching. |

### 3.2 Knowledge Service (`knowledge.proto`)

* **Owner:** .NET Service
* **Role:** The "Librarian" – Stateful data management.
* **Design Constraint:** Must support "Hybrid Search" (combining semantic vector similarity with exact metadata filtering).

#### **RPC Methods**

| Method | Input | Output | Purpose |
| --- | --- | --- | --- |
| `IngestChunks` | `IngestRequest` (List of Chunks) | `IngestResponse` (Success/Fail) | Bulk-saves analyzed text and vectors into `pgvector`. |
| `SearchSimilar` | `SearchRequest` (Vector + Filters) | `SearchResponse` (Relevant Context) | Finds the top-k most similar text snippets to a query vector. |

---

## 4. Data Model Design (The "Nouns")

Understanding the data structures is crucial for implementation.

### **4.1 The `Chunk`**

The atomic unit of information. We do not store whole documents; we store "Chunks".

* **Definition:** A segment of text (e.g., 500 words) that represents a single coherent thought.
* **Fields:**
* `text_content` (string): The actual human-readable text.
* `vector` (repeated float): The machine-readable meaning (e.g., `[0.12, -0.98, ...]`).
* `source_page` (int): Metadata for citation (e.g., "Found on page 3").

### **4.2 The `Vector`**

* **Definition:** An array of floating-point numbers representing semantic meaning.
* **Design Note:** We use `repeated float` in Protobuf. For the Gemini model, this will typically be an array of length 768 or 1536.

---

## 5. Implementation Roadmap (Contract Layer)

This documentation governs the creation of the `.proto` files.

1. **Draft:** Write `intelligence.proto` and `knowledge.proto` (Completed in design).
2. **Generate:** Use `protoc` (The Protocol Buffers Compiler) to generate:
    * `*.pb.go` (Go Code)
    * `*_pb2.py` (Python Code)
    * `*.cs` (C# Code)


3. **Distribute:** Place the generated code into the respective service folders (`gateway-go/internal/gen`, `ai-service/gen`, etc.).
