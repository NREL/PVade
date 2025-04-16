import numpy as np
import pandas as pd
import os
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.stats import gaussian_kde
from time import time

class PVadeH5File():
    def __init__(self, h5_file, stream_rows, stream_spacing, table_chord, span_rows, span_spacing, table_span, elevation, tracker_angle, wind_direction, n_panels, start_time=2.0):

        self.h5_file = h5_file
        self.n_stream_rows = int(stream_rows)
        self.stream_spacing = stream_spacing
        self.table_chord = table_chord
        self.n_span_rows = int(span_rows)
        self.span_spacing = span_spacing
        self.table_span = table_span
        self.elevation = elevation
        self.tracker_angle = tracker_angle
        self.wind_direction = wind_direction
        self.nx_panels, self.ny_panels = n_panels

        self.x_centers = np.array([i*self.stream_spacing for i in range(self.n_stream_rows)])
        print("x_centers: ", self.x_centers)
        self.x_tables = np.array([[center-self.table_chord/2, center+self.table_chord/2] for center in self.x_centers])
        print("x_corners: ", self.x_tables)

        self.y_centers = np.array([i*self.span_spacing for i in range(self.n_span_rows)]) - (self.n_span_rows-1)/2*self.span_spacing
        print("y_centers: ", self.y_centers)
        self.y_tables = np.array([[center-self.table_span/2, center+self.table_span/2] for center in self.y_centers])
        print("y_corners: ", self.y_tables)

        self.table_extent = np.array([[-self.table_chord/2, self.table_chord/2],
                                      [-self.table_span/2,  self.table_span/2]])

        self.table_keys = [f'{i}_{j}' for i in range(self.n_stream_rows) for j in range(self.n_span_rows)]

        with h5py.File(h5_file, 'r') as f:

            # # get 3d mesh in format (n_points, 3)
            xyz_mesh = f['Mesh/structure_mesh.xdmf/geometry'][:, :]

            deformation_timeseries = f['Function/Deformation']
            self.timesteps_dict = {self.string_to_float(s): s for s in deformation_timeseries.keys()}
            self.timesteps_float = sorted(self.timesteps_dict.keys())
            self.time = np.array(self.timesteps_float)[np.where(np.array(self.timesteps_float)>=start_time)]
            print(f"\nFinal time: {self.timesteps_float[-1]}, with timestep: {self.timesteps_float[1] -  self.timesteps_float[0]}")
            print(f"Number of timesteps: {len(self.timesteps_float)}")
            

        #########################################
        # 1. Remove wind_direction rotation
        if wind_direction != 270:
            xyz_mesh = self.remove_wind_direction_rotation(xyz_mesh)
        #########################################
        # 2. Split into panel tables
        self.ind_tables, xyz_tables = self.split_into_tables(xyz_mesh)
        #########################################
        # 3. remove tracker angle
        self.xyz_tables = self.remove_tracker_angle(xyz_tables)
        #########################################
        # 4. split into upper and lower surface
        self.ind_table_up, self.ind_table_lo, self.xy_table_up, self.xy_table_lo = self.split_upper_lower_surface(self.xyz_tables, self.ind_tables)
        # make interpolation mesh for panels
        self.xy_panels = self.make_panel_mesh()

    def remove_wind_direction_rotation(self, xyz):
        # shift to the center of mass of the PV array
        center_of_array = (np.sum(self.x_centers)/self.n_stream_rows, 
                           np.sum(self.y_centers)/self.n_span_rows, 
                           self.elevation)
        xyz_centered = xyz - center_of_array
        # rotate
        rotation = Rotation.from_euler('z', 270 - self.wind_direction, degrees=True)
        xyz_rotated = rotation.apply(xyz_centered)

        return xyz_rotated + center_of_array
    
    def split_into_tables(self, xyz):
        """ Split mesh coordinates into separate tables of PV panels. 
            Return two dictionaries with keys "i_j" for a table in i-th row and j-th column 
            (first dictionary with indices of mesh coordinates, second with mesh coordinates).

        :param xyz: mesh coordinates np.array of shape (n_points, 3)
        :return: dictionary of indicies of panel tables mesh points, dictionary of panel tables mesh points
        """
        eps = 0.1
        ind_tables, xyz_tables = {}, {}
        for i, x_table in enumerate(self.x_tables):
            for j, y_table in enumerate(self.y_tables):
                ind_tables[f'{i}_{j}'] = np.where((xyz[:, 0]>x_table[0]-eps) & (xyz[:, 0]<x_table[1]+eps) & (xyz[:, 1]>y_table[0]-eps) & (xyz[:, 1]<y_table[1]+eps))[0]
                xyz_tables[f'{i}_{j}'] = xyz[ind_tables[f'{i}_{j}']]
        print("split_into_panels:")
        for key, item in ind_tables.items():
            print(f"{key}:, {item.shape}")
        return ind_tables, xyz_tables

    def remove_tracker_angle(self, xyz_tables):
        """Rotates panel tables in x-z (stream-elevation) direction to remove tracker angle.

        :param xyz_tables: dictionary of panel tables mesh
        :return: dictionary of panel tables mesh points without tracker angle rotation
        """
        flat_tables = {}
        for i, x_c in enumerate(self.x_centers):
            for j, y_c in enumerate(self.y_centers):
                xyz = xyz_tables[f'{i}_{j}']
                xyz -= np.array([x_c, y_c, self.elevation])
                rotation = Rotation.from_euler('y', -self.tracker_angle, degrees=True)
                flat_tables[f'{i}_{j}'] = rotation.apply(xyz)
        return flat_tables
    

    def split_upper_lower_surface(self, flat_tables, ind_tables):
        """_summary_

        :param flat_tables: dict of xyz mesh coordinates for each panel table
        :param ind_tables: dict of mesh points indices for each panel table
        :return: dict of xy mesh coordinates and mesh points indices beloning to upper and lower surface
        """
        upper_xy, lower_xy = {}, {}
        upper_ind, lower_ind = {}, {}
        for key, xyz in flat_tables.items():
            ind_up = np.where(xyz[:, 2] > 0)[0]
            print(f"{key}: upper surface {ind_up.shape}")
            upper_ind[key] = ind_tables[key][ind_up]
            upper_xy[key] = xyz[ind_up][:, :2]

            ind_lo = np.where(xyz[:, 2] < 0)[0]
            print(f"{key}: lower surface {ind_lo.shape}")
            lower_ind[key] = ind_tables[key][ind_lo]
            lower_xy[key] = xyz[ind_lo][:, :2]
        return upper_ind, lower_ind, upper_xy, lower_xy


    def make_interpolator(self, data, data_type, resolution=10):

        # Define the grid over which to interpolate
        eps = 1e-8      # make sure that x, y are inside of interpolation domain
        x = np.linspace(self.table_extent[0, 0]+eps, self.table_extent[0, 1]-eps, int(resolution*self.table_chord))
        y = np.linspace(self.table_extent[1, 0]+eps, self.table_extent[1, 1]-eps, int(resolution*self.table_span))
        X, Y = np.meshgrid(x, y, indexing='ij')

        interpolator_dict = {}
        for key in self.table_keys:
            if data_type == 'deformation':
                deformation_magnitude = np.linalg.norm(data, axis=1)
                deformation_upper = LinearNDInterpolator(self.xy_table_up[key], deformation_magnitude[self.ind_table_up[key]])(X, Y)
                deformation_lower = LinearNDInterpolator(self.xy_table_lo[key], deformation_magnitude[self.ind_table_lo[key]])(X, Y)
                values = (deformation_upper + deformation_lower)/2
                
            elif data_type == 'stress':
                normal_vector = self.get_surface_normal()  # normal out of the upper surface
                stress_upper = data[self.ind_table_up[key]]
                traction_upper = (stress_upper.reshape((-1, 3, 3)) @ normal_vector) #@ normal_vector
                traction_upper = LinearNDInterpolator(self.xy_table_up[key], traction_upper)(X, Y)
                stress_lower = data[self.ind_table_lo[key]]
                traction_lower = (stress_lower.reshape((-1, 3, 3)) @ normal_vector) #@ normal_vector
                traction_lower = LinearNDInterpolator(self.xy_table_lo[key], traction_lower)(X, Y)
                traction = traction_upper + traction_lower
                values = traction @ normal_vector
            
            interpolator_dict[key] = RegularGridInterpolator((x, y), values, method='linear')

        return interpolator_dict


    def interpolate_table_data(self, dict_interpolator, resolution=10):

        # Define the grid over which to interpolate
        eps = 1e-7      # make sure that x, y are inside of interpolation domain
        x = np.linspace(self.table_extent[0, 0]+eps, self.table_extent[0, 1]-eps, int(resolution*self.table_chord))
        y = np.linspace(self.table_extent[1, 0]+eps, self.table_extent[1, 1]-eps, int(resolution*self.table_span))
        X, Y = np.meshgrid(x, y, indexing='ij')

        interpolated_data = {}
        for key, interpolator in dict_interpolator.items():
            interpolated_data[key] = interpolator((X, Y))
        return X, Y, interpolated_data


    def get_point_data_timeseries(self, xy_frac, data_timeseries, data_type):

        X = xy_frac[0] * self.table_chord - self.table_chord/2
        Y = xy_frac[1] * self.table_span - self.table_span/2

        point_data = {k: [] for k in self.table_keys}
    
        for i, t in enumerate(self.time):
            if i%50==0:
                print(f"t={t}")
            data_t = data_timeseries[self.timesteps_dict[t]]
            interpolator = self.make_interpolator(data_t, data_type)
            for key in self.table_keys:
                point_data[key].append(interpolator[key]((X, Y)))
        return X, Y, point_data

    # TODO: make a better resolution definition
    def make_panel_mesh(self, resolution=10):
        """
        :param resolution: _description_, defaults to 10
        """
        self.panel_chord = self.table_chord/(self.nx_panels)
        self.panel_span = self.table_span/(self.ny_panels)

        x_points = int(resolution*self.panel_chord)
        y_points = int(resolution*self.panel_span)

        xy_panels = [[None] * self.ny_panels for _ in range(self.nx_panels)]
        # counting panels from the top left corner (for easier plotting)
        eps = 1e-7
        for i in range(self.nx_panels):
            for j in range(self.ny_panels):
                x = np.linspace(-self.table_chord/2+eps, -self.table_chord/2 + self.panel_chord-eps, x_points) + i*self.panel_chord
                y = np.linspace(self.table_span/2 - self.panel_span+eps, self.table_span/2-eps, y_points) - j*self.panel_span      
                X, Y = np.meshgrid(x, y, indexing='ij')
                xy_panels[i][j] = (X, Y)
        return xy_panels

    def get_panels_data(self, table_interpolator, scalar_value=True):
        panels_data = {}
        panels_data_shape = self.xy_panels[0][0][0].shape
        for key, interpolator in table_interpolator.items(): 
            
            key_data = np.empty((self.nx_panels, self.ny_panels))
            if not scalar_value: 
                key_data = np.empty((self.nx_panels, self.ny_panels, panels_data_shape[0], panels_data_shape[1]))

            for i, y_panels in enumerate(self.xy_panels):
                for j, mesh in enumerate(y_panels):
                    data = interpolator((mesh))
                    if scalar_value:
                        key_data[i, j] = np.max(data)
                    else:
                        key_data[i, j] = data

            panels_data[key] = key_data
        return panels_data


    def get_panel_data_timeseries(self, data_timeseries, data_type):

        panels_timeseries = {k: np.empty((self.nx_panels, self.ny_panels, len(self.time))) for k in self.table_keys}
        for i, t in enumerate(self.time):
            if i%50==0:
                print(f"t={t}")
            data_t = data_timeseries[self.timesteps_dict[t]]
            table_interpolator = self.make_interpolator(data_t, data_type)
            panels_data_t = self.get_panels_data(table_interpolator, scalar_value=True)
            for key in self.table_keys:
                panels_timeseries[key][:, :, i] = panels_data_t[key]

        return panels_timeseries

    def get_surface_normal(self):
        n = [0, 0, 1] 
        rotation = Rotation.from_euler('y', self.tracker_angle, degrees=True)
        n = rotation.apply(n)
        return n

    @staticmethod
    def string_to_float(s):
        return float(s.replace("_", "."))





if __name__ == "__main__":

    # x-direction
    stream_rows = 7
    stream_spacing = 10.0
    panel_chord = 4.1

    # y-direction
    span_rows = 3
    span_spacing = 30.0
    panel_span = 24.25

    # z-direction
    elevation = 2.1
    panel_thickness = 0.1

    wind_direction = 270
    tracker_angle = 40

    ye = 400000000
    wind_speed = 16

    n_panels = (2, 23)
    start_time = 2.0

    solution_folder = f"/projects/solarflow/multirow_output/ye_{ye}/windspeed_{wind_speed:02d}/angle_{tracker_angle:02}/"
    h5_file = os.path.join(solution_folder, "solution", "solution_structure.h5")
    output_folder = f'/projects/solarflow/odoronin/plots/ws{wind_speed:02d}_angle{tracker_angle:02}/'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'deformation'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'stress'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'csv_data'), exist_ok=True)

    #### prints the whole structure of the file
    with h5py.File(h5_file, 'r') as f:
        for k in f['Function'].keys():
          print(k)

    data = PVadeH5File(h5_file, stream_rows, stream_spacing, panel_chord, 
                                span_rows, span_spacing, panel_span, 
                                elevation, tracker_angle, wind_direction,
                                n_panels, start_time)

    with h5py.File(h5_file, 'r') as f:
        deformation_last = f['Function/Deformation'][data.timesteps_dict[data.timesteps_float[-1]]] #(n_points, 3)
        xyz_mesh = f['Mesh/structure_mesh.xdmf/geometry']

        deformation_magnitude_last = np.linalg.norm(deformation_last, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sc = ax.scatter(xyz_mesh[:, 0], xyz_mesh[:, 1], c=deformation_magnitude_last)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(sc, label='deformation magnitude')
        fig.savefig(os.path.join(output_folder, 'deformation', 'deformation_last.png'))
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sc = ax.scatter(xyz_mesh[:, 0], xyz_mesh[:, 2], c=deformation_magnitude_last)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        plt.colorbar(sc, label='deformation magnitude')
        fig.savefig(os.path.join(output_folder, 'deformation','deformation_last_z.png'))
        plt.close()


    ##5. reinterpolate panel tables data 
    ## Deformation

    with h5py.File(h5_file, 'r') as f:

        deformation_last = f['Function/Deformation'][data.timesteps_dict[data.timesteps_float[-1]]]       # (n_points, 3)
        deformation_magnitude_last = np.linalg.norm(deformation_last, axis=1)
        deform_interpolator = data.make_interpolator(deformation_last, 'deformation')
        X, Y, interpolated_deformation = data.interpolate_table_data(deform_interpolator)

        for key, ind in data.ind_table_up.items():
            fig, ax = plt.subplots(1, 2, figsize=(8, 8))
            x, y = data.xy_table_up[key][:, 0], data.xy_table_up[key][:, 1]
            sc = ax[0].scatter(x, y, c=deformation_magnitude_last[ind])
            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[0].axis(data.table_extent.flatten())
            plt.colorbar(sc, label='deformation magnitude')
            # cax = ax[1].contourf(X, Y, interpolated_deformation[key])
            cax = ax[1].imshow(interpolated_deformation[key].T, extent=data.table_extent.flatten(), 
                               origin='lower', aspect='auto', cmap='viridis', 
                               vmin=0, vmax=np.max(deformation_magnitude_last))
            plt.colorbar(cax, label='deformation magnitude')
            fig.savefig(os.path.join(output_folder, 'deformation', f'deformation_last_{key}.png'))
            plt.close()

    # Stress

        stress_last = f['Function/stress_fluid'][data.timesteps_dict[data.timesteps_float[-1]]]
        stress_interpolator = data.make_interpolator(stress_last, 'stress')
        X, Y, interpolated_stress = data.interpolate_table_data(stress_interpolator)

        for key, ind in data.ind_table_up.items():
            fig, ax = plt.subplots(1, 1, figsize=(5, 8))
            x, y = data.xy_table_up[key][:, 0], data.xy_table_up[key][:, 1]
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis(data.table_extent.flatten())
            absmax = max(np.abs(np.min(interpolated_stress[key])), np.max(interpolated_stress[key]))
            cax = ax.imshow(interpolated_stress[key].T, extent=data.table_extent.flatten(), 
                            origin='lower', aspect='auto',
                            vmin=-absmax, vmax=absmax, cmap='coolwarm')
            plt.colorbar(cax, label='normal traction')
            fig.savefig(os.path.join(output_folder, 'stress', f'stress_last_{key}.png'))
            plt.close()

        #########################################
        # 6. get data for subpanels
        
        panel_mesh_shape = data.xy_panels[0][0][0].shape
        print("panel_mesh_shape", panel_mesh_shape)

        deformation_panels = data.get_panels_data(deform_interpolator, scalar_value=False)
        for key, data_table in deformation_panels.items():
            fig, ax = plt.subplots(n_panels[1], n_panels[0], figsize=(4, 8), sharey='row', sharex='col')
            for i, data_span in enumerate(data_table):
                for j, data_j in enumerate(data_span):
                    extent = (np.min(data.xy_panels[i][j][0]), np.max(data.xy_panels[i][j][0]), np.min(data.xy_panels[i][j][1]), np.max(data.xy_panels[i][j][1]))
                    cax = ax[j, i].imshow(data_j.T, extent=extent, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=np.max(deformation_magnitude_last))
                # plt.colorbar(cax)
            fig.savefig(os.path.join(output_folder, 'deformation', f'deformation_panels_{key}.png'))
            plt.close()

        stress_panels = data.get_panels_data(stress_interpolator, scalar_value=False)
        for key, data_table in stress_panels.items():
            fig, ax = plt.subplots(n_panels[1], n_panels[0], figsize=(4, 8), sharey='row', sharex='col')
            for i, data_span in enumerate(data_table):
                for j, data_j in enumerate(data_span):
                    extent = (np.min(data.xy_panels[i][j][0]), np.max(data.xy_panels[i][j][0]), np.min(data.xy_panels[i][j][1]), np.max(data.xy_panels[i][j][1]))
                    cax = ax[j, i].imshow(data_j.T, extent=extent, origin='lower', aspect='auto', vmin=-absmax, vmax=absmax, cmap='coolwarm')
                # plt.colorbar(cax)
            fig.savefig(os.path.join(output_folder, 'stress', f'stress_panels_{key}.png'))
            plt.close()

        #########################################
        ## 7. extract point timeseries (%chord, %span)
        xy_frac = (0.95, 0.05)   # x_frac, y_frac
        print(f"\n\nExtracting timeseries at: {xy_frac[0]} chord, {xy_frac[1]} span")
        # Deformation
        print("Deformation")
        t1 = time()
        deformation_timeseries = f['Function/Deformation']
        _, _, values = data.get_point_data_timeseries(xy_frac, deformation_timeseries, 'deformation') # ~505s
        print("Time: ", time()-t1)  

        fig, ax = plt.subplots(data.n_stream_rows, 1, figsize=(10, 16), sharex='col', sharey='col')
        for name, v in values.items():
            row = int(name[0])
            ax[row].plot(data.time, v, '-', label=name)
        for row in range(data.n_stream_rows):
            ax[row].legend()
            ax[row].set_ylabel('Deformation magnitude')
        ax[-1].set_xlabel('time')
        
        ax[0].set_title(f"Location: {xy_frac[0]} chord, {xy_frac[1]} span")
        fig.savefig(os.path.join(output_folder, 'deform_time.png'), bbox_inches='tight')
        plt.close()

        print("Stress")
        t1 = time()
        stress_timeseries = f['Function/stress_fluid']
        _, _, values = data.get_point_data_timeseries(xy_frac, stress_timeseries, data_type='stress')
        print("Time: ", time()-t1)  
        fig, ax = plt.subplots(data.n_stream_rows, 1, figsize=(10, 16), sharex='col', sharey='col')
        for name, v in values.items():
            row = int(name[0])
            ax[row].plot(data.time, v, '-', label=name)
        for row in range(data.n_stream_rows):
            ax[row].legend()
            ax[row].set_ylabel('normal traction')
        ax[-1].set_xlabel('time')
        
        ax[0].set_title(f"Location: {xy_frac[0]} chord, {xy_frac[1]} span")
        fig.savefig(os.path.join(output_folder, 'stress_time.png'), bbox_inches='tight')
        plt.close()

        # #########################################
    # 8. get surrogate output: max stress/deformation for every panel
    with h5py.File(h5_file, 'r') as f:   
        deformation_timeseries = f['Function/Deformation']
        # t1 = time()
        # panel_data_timeseries = data.get_panel_data_timeseries(deformation_timeseries, data_type='deformation')
        # np.savez(os.path.join(output_folder, 'panel_deformation_timeseries.npz'), **panel_data_timeseries)
        # print("Time: ", time()-t1) 
        panel_data_timeseries = np.load(os.path.join(output_folder, 'panel_deformation_timeseries.npz'))

        fig, ax = plt.subplots(data.n_stream_rows, data.n_span_rows, figsize=(20, 20), sharex='all', sharey='all')
        for name, table_data in panel_data_timeseries.items():
            row = int(name[0])
            col = int(name[2])
            for i, span in enumerate(table_data):
                for j, value in enumerate(span):
                    ax[row, col].plot(data.time, value, '-', label=f"panel {i}_{j}")
            ax[row, col].set_title(f"table {name}")
        for row in range(data.n_stream_rows):
            ax[row, 0].set_ylabel('max deformation magnitude')
        for col in range(data.n_span_rows):
            ax[-1, col].set_xlabel('time')
        plt.legend(bbox_to_anchor=(0.5, 0.9), bbox_transform=fig.transFigure, loc="lower center", borderaxespad=0, ncol=n_panels[0]*n_panels[1])
        plt.subplots_adjust(top=0.88, wspace=0.05, hspace=None)
        fig.savefig(os.path.join(output_folder, 'deformation_timeseries_by_table.png'), bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(4, 23, figsize=(30, 12), sharex='row', sharey='row')
        for name, table_data in panel_data_timeseries.items():
            row = int(name[0])
            col = int(name[2])
            if row > 2:
                for i, span in enumerate(table_data):
                    for j, value in enumerate(span):
                        ax[i, j].plot(data.time, value, '-', label=f"table {name}")
                        ax[i, j].set_title(f"panel {i}_{j}")
                        d_range = np.linspace(0, 0.15)
                        kde = gaussian_kde(value)
                        ax[i+2, j].plot(d_range, kde(d_range), '-', label=f"table {name}")
                        ax[i+2, j].set_title(f"panel {i}_{j}")
                        ax[i+2, j].axis(xmin=0, xmax=0.15)
        for row in range(n_panels[0]):
            ax[row, 0].set_ylabel('max deformation magnitude')
            for col in range(n_panels[1]):
                ax[row, col].set_xlabel('time')
        for row in range(n_panels[0]):
            ax[row+2, 0].set_ylabel('distribution')
            for col in range(n_panels[1]):
                ax[row+2, col].set_xlabel('max deformation magnitude')
        plt.legend(bbox_to_anchor=(0.5, 0.9), bbox_transform=fig.transFigure, loc="lower center", borderaxespad=0, ncol=data.n_stream_rows-3)
        plt.subplots_adjust(top=0.85, wspace=0.05, hspace=0.4)
        fig.savefig(os.path.join(output_folder, 'deformation_timeseries_by_panel.png'), bbox_inches='tight')
        plt.close()


        stress_timeseries = f['Function/stress_fluid']
        # t1 = time()
        # panel_data_timeseries = data.get_panel_data_timeseries(stress_timeseries, data_type='stress')
        # np.savez(os.path.join(output_folder, 'panel_stress_timeseries.npz'), **panel_data_timeseries)
        # print("Time: ", time()-t1) 
        panel_data_timeseries = np.load(os.path.join(output_folder, 'panel_stress_timeseries.npz'))


        fig, ax = plt.subplots(data.n_stream_rows, data.n_span_rows, figsize=(20, 20), sharex='all', sharey='all')
        for name, table_data in panel_data_timeseries.items():
            row = int(name[0])
            col = int(name[2])
            for i, span in enumerate(table_data):
                for j, value in enumerate(span):
                    ax[row, col].plot(data.time, value, '-', label=f"panel {i}_{j}")
            ax[row, col].set_title(f"table {name}")
        for row in range(data.n_stream_rows):
            ax[row, 0].set_ylabel('max normal stress')
        for col in range(data.n_span_rows):
            ax[-1, col].set_xlabel('time')
        plt.legend(bbox_to_anchor=(0.5, 0.9), bbox_transform=fig.transFigure, loc="lower center", borderaxespad=0, ncol=n_panels[0]*n_panels[1])
        plt.subplots_adjust(top=0.88, wspace=0.05, hspace=None)
        fig.savefig(os.path.join(output_folder, 'stress_timeseries_by_table.png'), bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(4, 23, figsize=(30, 12), sharex='row', sharey='row')
        for name, table_data in panel_data_timeseries.items():
            row = int(name[0])
            col = int(name[2])
            if row > 2:
                for i, span in enumerate(table_data):
                    for j, value in enumerate(span):
                        ax[i, j].plot(data.time, value, '-', label=f"table {name}")
                        ax[i, j].set_title(f"panel {i}_{j}")
                        d_range = np.linspace(0, 80)
                        kde = gaussian_kde(value)
                        ax[i+2, j].plot(d_range, kde(d_range), '-', label=f"table {name}")
                        ax[i+2, j].set_title(f"panel {i}_{j}")
                        ax[i+2, j].axis(xmin=0, xmax=80)
        for row in range(n_panels[0]):
            ax[row, 0].set_ylabel('max normal stress')
            for col in range(n_panels[1]):
                ax[row, col].set_xlabel('time')
        for row in range(n_panels[0]):
            ax[row+2, 0].set_ylabel('distribution')
            for col in range(n_panels[1]):
                ax[row+2, col].set_xlabel('max normal stress')
        plt.legend(bbox_to_anchor=(0.5, 0.9), bbox_transform=fig.transFigure, loc="lower center", borderaxespad=0, ncol=data.n_stream_rows-3)
        plt.subplots_adjust(top=0.85, wspace=0.05, hspace=0.4)
        fig.savefig(os.path.join(output_folder, 'stress_timeseries_by_panel.png'), bbox_inches='tight')
        plt.close()

        print(data.get_surface_normal())

    # # 9. get single panel traction csv data
    # with h5py.File(h5_file, 'r') as f:  
    #     stress_timeseries = f['Function/stress_fluid']
    #     panels_timeseries = np.empty((data.nx_panels, data.ny_panels, len(data.time)))
    #     table_key = '6_0'
    #     i_panel, j_panel = 1, 4 
    #     X, Y = data.xy_panels[i_panel][j_panel]
    #     mesh = np.vstack([X.ravel(), Y.ravel()]).T
    #     for i, t in enumerate(data.time):
    #         if i%5==0:
    #             print('t=', t)
    #             data_t = stress_timeseries[data.timesteps_dict[t]]
    #             table_interpolator = data.make_interpolator(data_t, 'stress')
    #             panels_data_t = data.get_panels_data(table_interpolator, scalar_value=False)
    #             panel_data = panels_data_t[table_key][i_panel, j_panel]
    #             df_panel = pd.DataFrame(mesh, columns=['x', 'y'])
    #             df_panel['normal traction'] = panel_data.flatten()
    #             df_panel.to_csv(os.path.join(output_folder, 'csv_data', f'normal_traction_table_{table_key}_panel_{i_panel}{j_panel}_t_{data.timesteps_dict[t]}.csv'), index=False)
