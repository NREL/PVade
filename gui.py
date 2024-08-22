import sys
import yaml
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QScrollArea, QMainWindow, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QProcess

# Function to load the schema from input_schema.yaml
def load_schema(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to save the inputs.yaml file
def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

class YamlEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize QProcess for real-time output
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.on_process_finished)

        # Load input schema
        self.schema = load_schema('pvade/input_schema.yaml')

        # Initialize the data dictionary with default values
        self.data = {}
        self.load_defaults(self.schema, self.data)

        # Set up the GUI layout
        self.initUI()
        

    def load_defaults(self, schema, data):
        """
        Recursively load default values from the schema into the data dictionary.
        """
        for section, properties in schema.get('properties', {}).items():
            if properties.get('type') == 'object':
                data[section] = {}
                self.load_defaults(properties, data[section])
            else:
                data[section] = properties.get('default')

    def initUI(self):
        self.setWindowTitle('PVade Configuration')

        # Create a central widget and set it to a scroll area
        central_widget = QWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Picture on the left (resized to 1/4th of the original size)
        pixmap = QPixmap('pv_model.png')
        pixmap = pixmap.scaled(pixmap.width() // 4, pixmap.height() // 4)  # Resize the image
        picture_label = QLabel(self)
        picture_label.setPixmap(pixmap)
        main_layout.addWidget(picture_label)

        # Variable inputs on the right
        self.variables_layout = QVBoxLayout()
        self.inputs = {}

        # Dropdown menu for selecting examples
        self.example_dropdown = QComboBox(self)
        self.example_dropdown.addItems(['panel 2d', 'panel 3d', 'flag 2d', 'cylinder 2d', 'cylinder 3d'])
        self.example_dropdown.currentIndexChanged.connect(self.update_example)
        self.variables_layout.addWidget(self.example_dropdown)
        
        # Dynamically create input fields based on the schema
        for section, properties in self.schema.get('properties', {}).items():
            section_label = QLabel(f"<b>{section.capitalize()}</b>")
            self.variables_layout.addWidget(section_label)
            if properties.get('type') == 'object':
                self.create_inputs(properties, section)

        main_layout.addLayout(self.variables_layout)

        # Add the number of cores input
        cores_label = QLabel('Number of cores')
        self.number_of_cores_input = QLineEdit('1')  # Default value set to 1
        self.variables_layout.addWidget(cores_label)
        self.variables_layout.addWidget(self.number_of_cores_input)


        # Output text box at the bottom
        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        main_layout.addWidget(self.output_text)

        # Execute button
        execute_button = QPushButton('Execute', self)
        execute_button.clicked.connect(self.execute_script)
        self.variables_layout.addWidget(execute_button)

    def create_inputs(self, properties, section):
        """
        Create input fields based on the schema properties.
        """
        for var, prop in properties.get('properties', {}).items():
            hbox = QHBoxLayout()
            var_label = QLabel(var)
            if prop.get('type') == 'boolean':
                var_input = QComboBox()
                var_input.addItems(['True', 'False'])
                var_input.setCurrentIndex(0 if prop.get('default') else 1)
            else:
                var_input = QLineEdit(str(prop.get('default', '')))
            hbox.addWidget(var_label)
            hbox.addWidget(var_input)
            self.variables_layout.addLayout(hbox)
            self.inputs[f'{section}.{var}'] = var_input

    def update_example(self):
        example = self.example_dropdown.currentText()
        yaml_loc = "/Users/warsalan/work/duramat_pv/codes/PVade_pub2/input"
        yaml_file_mapping = {
            'panel 2d': yaml_loc+'/sim_params_2d.yaml',
            'panel 3d': yaml_loc+'/sim_params.yaml',
            'flag 2d': yaml_loc+'/flag2d.yaml',
            'cylinder 2d': yaml_loc+'/2d_cyld.yaml',
            'cylinder 3d': yaml_loc+'/3d_cyld.yaml'
        }
        file_path = yaml_file_mapping.get(example)

        if not file_path or not os.path.exists(file_path):
            self.output_text.append(f"Error: File '{file_path}' not found.")
            return

        with open(file_path, 'r') as file:
            example_data = yaml.safe_load(file)

        # Sort the variables according to the example data and update the GUI
        self.update_inputs_from_example(example_data)

    def update_inputs_from_example(self, example_data):
        """
        Update the GUI input fields with values from the selected example.
        Reset input fields not found in the example data to an empty string.
        """
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened_data = flatten_dict(example_data)
        
        # Update fields with data from example
        for key, value in flattened_data.items():
            if key in self.inputs:
                input_field = self.inputs[key]
                if isinstance(input_field, QComboBox):
                    input_field.setCurrentIndex(0 if value else 1)
                else:
                    input_field.setText(str(value))

        # Reset fields not found in the example data
        for key, input_field in self.inputs.items():
            if key not in flattened_data:
                if isinstance(input_field, QComboBox):
                    input_field.setCurrentIndex(1)  # Assuming False or the second item
                else:
                    input_field.setText("")

    

    def execute_script(self):
        # Clear the output console
        self.output_text.clear()

        # Scroll up to the top
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().minimum())
        def clean_data(data):
            """
            Recursively remove keys with null or empty values from the data dictionary.
            """
            if isinstance(data, dict):
                keys_to_remove = []
                for key, value in data.items():
                    cleaned_value = clean_data(value)  # Recursively clean nested dictionaries
                    if cleaned_value is None or cleaned_value == '':
                        keys_to_remove.append(key)
                    else:
                        data[key] = cleaned_value
                for key in keys_to_remove:
                    del data[key]
            elif isinstance(data, list):
                data = [clean_data(item) for item in data if clean_data(item) not in (None, '')]
            else:
                return data if data not in (None, '') else None
            return data
    
        # Update self.data with the input values
        for key, line_edit in self.inputs.items():
            section, var = key.split('.')
            if section not in self.data:
                self.data[section] = {}
            
            if isinstance(line_edit, QComboBox):
                value = line_edit.currentText() == 'True'
            else:
                value = yaml.safe_load(line_edit.text())
            
            # Only add to self.data if value is not empty
            if value or (not isinstance(value, str) and value is not None):
                self.data[section][var] = value
            else: 
                self.data[section][var] = ""
        
        # Clean the data by removing null or empty values
        self.data = clean_data(self.data)
        
        # Save the updated data to inputs.yaml
        save_yaml(self.data, 'inputs.yaml')
        # print(int(self.number_of_cores_input.text()))

        num_cores = int(self.number_of_cores_input.text())
        self.process.start('mpirun', ['-n', str(num_cores), 'python', 'ns_main.py', '--input', 'inputs.yaml'])
        # # Execute the external Python script and capture its output
        # process = subprocess.Popen(['mpirun', '-n', str(num_cores), 'python', 'ns_main.py', '--input', 'inputs.yaml'],
        #                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # stdout, stderr = process.communicate()

        # # Display the script output in the text box
        # self.output_text.append(f"Output:\n{stdout}")
        # if stderr:
        #     self.output_text.append(f"Errors:\n{stderr}")

        # # Scroll to the top of the output text area
        # cursor = self.output_text.textCursor()
        # cursor.movePosition(cursor.Start)
        # self.output_text.setTextCursor(cursor)

    def handle_stdout(self):
        # Append standard output to the output text box
        data = self.process.readAllStandardOutput().data().decode()
        self.output_text.append(data)
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def handle_stderr(self):
        # Append standard error to the output text box
        data = self.process.readAllStandardError().data().decode()
        self.output_text.append(f"Error: {data}")
        self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

    def on_process_finished(self):
        # Process finished, handle any final updates if needed
        self.output_text.append("Process finished.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YamlEditor()
    ex.show()
    sys.exit(app.exec_())
