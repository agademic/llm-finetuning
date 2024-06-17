import streamlit as st
import sqlite3
import json

# Connect to SQLite database
conn = sqlite3.connect('inputs.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS inputs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              instruction TEXT,
              input TEXT,
              output TEXT)''')


# Define function to view and delete entries
def view_entries():
    st.title("View Entries")

    # Select all entries from database
    c.execute("SELECT * FROM inputs")
    data = c.fetchall()

    # Display entries in a table
    if len(data) > 0:
        st.write("All Entries")
        for row in data:
            st.write("ID:", row[0])
            st.write("Instruction:", row[1])
            st.write("Input:", row[2])
            st.write("Output:", row[3])
            st.write("---")

        # Delete entry button
        id_to_delete = st.number_input("Enter the ID of the entry to delete:", value=1, min_value=1)
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Delete Entry"):
                c.execute("DELETE FROM inputs WHERE id=?", (id_to_delete,))
                conn.commit()
                st.success("Entry deleted successfully!")
        # with col2:
        #     if st.button("Delete All"):
        #         # Delete all entries from SQLite database
        #         c.execute("DELETE FROM inputs")
        #         conn.commit()
        with col3:
            extract_database()

    else:
        st.write("No entries found in database.")


def extract_database():
    # Export button
    if st.button("Export as JSON"):
        # Get all input data from SQLite database
        c.execute("SELECT * FROM inputs")
        data = c.fetchall()
        conn.close()

        # Convert data to JSON format
        json_data = []
        for row in data:
            json_data.append({
                # 'id': row[0],
                'instruction': row[1],
                'input': row[2],
                'output': row[3]
            })

        # Download JSON file
        with open('custom_instructions_data.json', 'w') as f:
            json.dump(json_data, f, indent=4)
        st.success("Data exported as JSON!")


# Define Streamlit app
def app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Input Data", "View Entries"])

    if page == "Input Data":
        st.title("Custom Instruction Database")
        st.header("Input Data")

        with st.form(key="Custom Instruction", clear_on_submit=True):
            # Text input fields
            instruction = st.text_input("Instruction")
            input_text = st.text_area("Input")
            output_text = st.text_area("Output")
            submit_button = st.form_submit_button("Save")

        # Save button
        if submit_button:
            # Insert input data into SQLite database
            c.execute('''INSERT INTO inputs (instruction, input, output)
                         VALUES (?, ?, ?)''', (instruction, input_text, output_text))
            conn.commit()
            st.success("Data saved successfully!")

    elif page == "View Entries":
        view_entries()

    # Close database connection
    conn.close()


# Run the Streamlit app
if __name__ == '__main__':
    app()
