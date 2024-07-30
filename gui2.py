import tkinter as tk
from tkinter import filedialog
import os
import subprocess
from tkhtmlview import HTMLLabel
# from tkinter.ttk import Notebook, Frame, Style, WebBrowserTab
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import subprocess
import os
import tempfile
class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("PVade")
        self.geometry("600x550")

        # Load preset values from the text file
        self.preset_values = self.load_preset_values("sample_org.yaml")
        # self.preset_values = self.load_preset_values("input/sim_params.yaml")

        self.categories = {
            "Introduction": [],
            # "Introduction": [
            #     {"label": "Test Case", "entry": None},
            # ],
            "general": [
                {"label": "test", "entry": None},
                {"label": "output_dir", "entry": None},
                {"label": "geometry_module", "entry": None},
                {"label": "input_mesh_dir", "entry": None},
                {"label": "mesh_only", "entry": None},
                {"label": "debug_mesh_motion_only", "entry": None},
                {"label": "structural_analysis", "entry": None},
                {"label": "fluid_analysis", "entry": None}
            ],
            "domain": [
                {"label": "x_min", "entry": None},
                {"label": "x_max", "entry": None},
                {"label": "y_min", "entry": None},
                {"label": "y_max", "entry": None},
                {"label": "z_min", "entry": None},
                {"label": "z_max", "entry": None},
                {"label": "l_char", "entry": None},
                {"label": "free_slip_along_walls", "entry": None}
            ],
            "pv_array": [
                {"label": "stream_rows", "entry": None},
                {"label": "span_rows", "entry": None},
                {"label": "elevation", "entry": None},
                {"label": "stream_spacing", "entry": None},
                {"label": "span_spacing", "entry": None},
                {"label": "span_fixation_pts", "entry": None},
                {"label": "panel_chord", "entry": None},
                {"label": "panel_span", "entry": None},
                {"label": "panel_thickness", "entry": None},
                {"label": "tracker_angle", "entry": None}
            ],
            "solver": [
                {"label": "dt", "entry": None},
                {"label": "t_final", "entry": None},
                {"label": "save_text_interval", "entry": None},
                {"label": "save_xdmf_interval", "entry": None},
                {"label": "solver1_ksp", "entry": None},
                {"label": "solver2_ksp", "entry": None},
                {"label": "solver3_ksp", "entry": None},
                {"label": "solver4_ksp", "entry": None},
                {"label": "solver1_pc", "entry": None},
                {"label": "solver2_pc", "entry": None},
                {"label": "solver3_pc", "entry": None},
                {"label": "solver4_pc", "entry": None}
            ],
            "fluid": [
                {"label": "u_ref", "entry": None},
                {"label": "initialize_with_inflow_bc", "entry": None},
                {"label": "time_varying_inflow_bc", "entry": None},
                {"label": "rho", "entry": None},
                {"label": "wind_direction", "entry": None},
                {"label": "nu", "entry": None},
                {"label": "dpdx", "entry": None},
                {"label": "turbulence_model", "entry": None},
                {"label": "c_s", "entry": None},
                {"label": "c_w", "entry": None},
                {"label": "bc_y_min", "entry": None},
                {"label": "bc_y_max", "entry": None},
                {"label": "bc_z_min", "entry": None},
                {"label": "bc_z_max", "entry": None},
                {"label": "periodic", "entry": None},
                {"label": "warm_up_time", "entry": None}
            ],
            "structure": [
                {"label": "beta_relaxation", "entry": None},
                {"label": "tube_connection", "entry": None},
                {"label": "motor_connection", "entry": None},
                {"label": "bc_list", "entry": None},
                {"label": "dt", "entry": None},
                {"label": "rho", "entry": None},
                {"label": "elasticity_modulus", "entry": None},
                {"label": "poissons_ratio", "entry": None},
                {"label": "body_force_x", "entry": None},
                {"label": "body_force_y", "entry": None},
                {"label": "body_force_z", "entry": None}
            ],
            "Execution": [
                {"label": "Number of Cores", "entry": None}
            ]
        }

        self.frames = {}
        self.current_frame = None
        self.index = 0

        self.create_frames()
        # self.show_frame("general")

        
        self.show_frame("Introduction")


    def load_preset_values(self, filename):
        preset_values = {}
        current_category = None
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.endswith(":"):
                    current_category = line[:-1]
                    preset_values[current_category] = {}
                elif current_category:
                    key, value = line.split(":", 1)
                    preset_values[current_category][key.strip()] = value.strip()
        return preset_values
    

    def create_frames(self):
        # self.frames["Introduction"]=self.create_intro_frame()
        for category in self.categories:
            self.frames[category] = self.create_category_frame(category, self.categories[category])

    def create_category_frame(self, category_name, entries):
        frame = tk.Frame(self, relief="groove", borderwidth=2)
        frame.focus_set()
        tk.Label(frame, text=category_name, font=("Helvetica", 12, "bold")).grid(row=0, columnspan=2)
        if category_name == "Introduction":
            
            # frame.pack(fill="both", expand=True)
            # Paragraph at the top
            paragraph = """PVade is an open source fluid-structure interaction model which can be used \n 
            to study wind loading and stability on solar-tracking PV arrays.  \n
            PVade can be used as part of a larger modeling chain to provide  \n 
            stressor inputs to mechanical module models to study the physics of failure \n 
            for degradation mechanisms such as cell cracking, \n
            weathering of cracked cells, and glass breakage. \n
            For more information, visit the PVade Documentation. \n
            https://pvade.readthedocs.io/en/latest/index.html"""
            tk.Label(frame, text=paragraph, font=("Helvetica", 12, "bold")).grid(row=0, columnspan=2)


            # # Add a WebBrowserTab with a web page
            # notebook = Notebook(frame)
            # notebook.grid(row=1, columnspan=2, pady=10)
            # web_browser_tab = WebBrowserTab(notebook)
            # notebook.add(web_browser_tab, text="PVade Website")
            # web_browser_tab.open("https://www.example.com")  # Replace with your desired URL


        else:    
       
            preset_values = self.preset_values.get(category_name, {})
            for i, entry in enumerate(entries):
                # if entry.get("toggle"):
                #     entry["toggle"] = tk.StringVar(value=preset_values.get(entry["label"], "Cylinder2D"))
                #     options = ["Cylinder2D", "Cylinder3D", "Flag2D", "Panels2D", "Panels3D"]
                #     tk.Label(frame, text=entry["label"] + ":").grid(row=i + 1, column=0, sticky="e")
                #     tk.OptionMenu(frame, entry["toggle"], *options).grid(row=i + 1, column=1, padx=5, pady=5)
                # else:
                entry_value = preset_values.get(entry["label"], "")
                tk.Label(frame, text=entry["label"] + ":").grid(row=i + 1, column=0, sticky="e")
                entry["entry"] = tk.Entry(frame, textvariable=tk.StringVar())
                entry["entry"].insert(0, entry_value)
                entry["entry"].grid(row=i + 1, column=1)

                

        if category_name != "Execution":
            next_button = tk.Button(frame, text="Next", command=lambda: self.next_frame(category_name))
            next_button.grid(row=len(entries) + 2, columnspan=2, pady=10)
        else:
            run_button = tk.Button(frame, text="Run", command=self.run)
            run_button.grid(row=len(entries) + 2, columnspan=2, pady=10)

            self.output_text = scrolledtext.ScrolledText(frame, width=80, height=30)
            self.output_text.grid(row=len(entries) + 3, columnspan=2, padx=10, pady=10)

               
        
        return frame

    def show_frame(self, category):
        if self.current_frame:
            self.current_frame.pack_forget()
        self.current_frame = self.frames[category]
        self.current_frame.pack(fill="both", expand=True)

    def next_frame(self, current_category):
        current_index = list(self.categories.keys()).index(current_category)
        next_index = current_index + 1
        if next_index >= len(self.categories):
            next_index = 0
        next_category = list(self.categories.keys())[next_index]
        self.show_frame(next_category)

    def run(self):
        filename = "input.yaml"
        with open(filename, 'w') as file:
            for category, entries in self.categories.items():
                if category != "Execution" and category != "Introduction" :
                    file.write(f"{category}:\n")
                    for entry in entries:
                        if entry.get("toggle") :
                            file.write(f"  {entry['label']}: {entry['toggle'].get()}\n")
                        else:
                            if entry['entry'].get() != "" :
                                file.write(f"  {entry['label']}: {entry['entry'].get()}\n")
                # file.write("\n")
        print("Data saved to", filename)
        
        num_cores = self.categories["Execution"][0]["entry"].get() or "4"
        command = ["mpirun", "-n", num_cores, "python", "ns_main.py", "--input", filename]
        # command = ["python", "ns_main.py"]#, "--input", filename]
        
        # Execute ns_main.py from the current directory
        current_dir = os.getcwd()
        print(command)
        # subprocess.run(command)
        current_dir = os.getcwd()

        # Clear previous output
        self.output_text.delete('1.0', tk.END)

        # Redirect subprocess output to the Text widget
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in iter(process.stdout.readline, ""):
            self.output_text.insert(tk.END, line)
            self.output_text.see(tk.END)  # Scroll to the end of the text

        process.stdout.close()
        process.wait()

if __name__ == "__main__":
    app = Application()
    app.mainloop()


