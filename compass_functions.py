import pandas as pd
import numpy as np
import calendar
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.text import TextPath
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
import warnings
warnings.filterwarnings("ignore")



def compass(r,T,s):
    return r*np.exp(1j*2*np.pi*s/T)

def init_atlas(sx,sy):
    
    plt.figure(figsize=(sx,sy))



def see_atlas(atlas,symbol_size,symbol_shape):
    
    plt.scatter(atlas.real,atlas.imag,s=symbol_size,marker=symbol_shape)


def color_atlas(atlas,symbol_size,symbol_shape,symbol_color,cmap,T_color):
    
    get_color =  cm.get_cmap(cmap, T_color)
    plt.scatter(atlas.real,atlas.imag,s=symbol_size,marker=symbol_shape,color=get_color(symbol_color%T_color))

def atlas_view(symbol,T_symbol,atlas,symbol_size,symbol_shape,symbol_color,cmap,T_color,char_flag,T_char,char_color,char_size,cmap_char):
    
    get_color =  cm.get_cmap(cmap, T_color)
    get_char_color = cm.get_cmap(cmap_char, T_char)
    plt.scatter(atlas.real,atlas.imag,s=symbol_size,marker=symbol_shape,color=get_color(symbol_color%T_color))
    if char_flag:
        plt.text(atlas.real,atlas.imag,str(symbol%T_symbol),size=char_size,color=get_char_color(char_color%T_symbol))



def compass_signed(r,T_pos,T_neg,s):
    

   return r*np.exp(1j*(np.pi/T_pos)*s) 

def get_colors(symbol,cmap,variable_sizes,base_size):
    
    markers = ['o' if t >= 0 else 'd' for t in symbol]
    norm = plt.Normalize(min(symbol), max(symbol))
    colors = [cmap(norm(value)) for value in symbol]
    norm_seq =norm(symbol)
    if variable_sizes:
        sizes=(base_size+1)*np.abs(symbol)
    else:
        sizes=base_size*np.ones(len(symbol))
    return  markers,norm,colors,norm_seq,sizes

def plot_atlas(atlas,symbol,cmap,alpha,sx,sy,legend_flag,variable_sizes,base_size):
    
    markers,norm,colors,norm_seq,sizes=get_colors(symbol,cmap,variable_sizes,base_size)
    
    init_atlas(sx,sy)
    for i in range(len(atlas)):
        plt.scatter(atlas[i].real, atlas[i].imag, color=colors[i], s=sizes[i], marker=markers[i], facecolor=colors[i], edgecolors='black', alpha=alpha)
    if legend_flag:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))

def plot_atlas_text(atlas,symbol,cmap,alpha,sx,sy,legend_flag,variable_sizes,base_size):
    markers,norm,colors,norm_seq,sizes=get_colors(symbol,cmap,variable_sizes,base_size)
    
    init_atlas(sx,sy)
    for i in range(len(atlas)):
        plt.scatter(atlas[i].real, atlas[i].imag, color=colors[i], s=sizes[i], marker=markers[i], facecolor=colors[i], edgecolors='black', alpha=alpha)
        plt.text(atlas[i].real, atlas[i].imag,str(symbol[i]),size=sizes[i],c=colors[i])
    if legend_flag:
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))

# Animations

def get_atlas_video(atlas_in, symbol, fps, colormap, output_path, variable_size, fixed_size,padding,sx,sy):
    
    
    
    
    markers, norm, colors, norm_seq, sizes = get_colors(symbol, colormap, variable_size, fixed_size)

    

    fig, ax = plt.subplots(figsize=(sx, sy))
    scatter = ax.scatter([], [], color=[], s=[], edgecolors='black', alpha=0.5)

    frames = len(atlas_in)
    x_values = []
    y_values = []
    sizes_values = []
    colors_values = []
    markers_values = []

    def update(frame):
        # Accumulate coordinates, sizes, colors, and markers from all previous frames
        x_values.extend([atlas_in[frame].real])
        y_values.extend([atlas_in[frame].imag])
        sizes_values.extend([sizes[frame]])
        colors_values.extend([colors[frame]])
        markers_values.extend([symbol[frame]])

        # Update the scatter plot with accumulated coordinates, sizes, colors, and markers
        scatter.set_offsets(np.column_stack((x_values, y_values)))
        scatter.set_sizes(sizes_values)
        scatter.set_color(colors_values)
        scatter.set_edgecolors('black')
        scatter.set_alpha(0.5)

        # Determine marker based on the value of the symbol for all frames
        marker_paths = []
        for marker in markers_values:
            if marker >= 0:
                marker_path = Path.unit_circle()
            else:
                marker_path = Path.unit_regular_polygon(4)
            marker_paths.append(marker_path)

        scatter.set_paths(marker_paths)

        # Adjust the axis limits based on the data
        ax.set_xlim(np.min(x_values) - padding, np.max(x_values) + padding)
        ax.set_ylim(np.min(y_values) - padding, np.max(y_values) + padding)

    animation = FuncAnimation(fig, update, frames=frames, blit=False)

    # Specify the writer (FFMpegWriter) and the output filename
    writer = FFMpegWriter(fps=fps)  # Adjust the frames per second (fps) as needed
    animation.save(output_path, writer=writer)


def get_atlas_video_text(atlas_in, symbol, fps, colormap, output_path, variable_size, fixed_size, padding, sx, sy, text_size, alpha):
    markers, norm, colors, norm_seq, sizes = get_colors(symbol, colormap, variable_size, fixed_size)

    fig, ax = plt.subplots(figsize=(sx, sy))
    scatter = ax.scatter([], [], color=[], s=[], edgecolors='black', alpha=alpha)

    frames = len(atlas_in)
    x_values = []
    y_values = []
    sizes_values = []
    colors_values = []
    markers_values = []

    def update(frame):
        x_values.clear()
        y_values.clear()
        sizes_values.clear()
        colors_values.clear()
        markers_values.clear()

        for i in range(frame + 1):
            x_values.append(atlas_in[i].real)
            y_values.append(atlas_in[i].imag)
            sizes_values.append(sizes[i])
            colors_values.append(colors[i])
            markers_values.append(markers[i])

        scatter.set_offsets(np.column_stack((x_values, y_values)))
        scatter.set_sizes(sizes_values)
        scatter.set_color(colors_values)
        scatter.set_edgecolors('black')
        scatter.set_alpha(alpha)

        marker_paths = []
        for marker in markers_values:
            if marker == 'o':
                marker_path = Path.unit_circle()
            else:
                marker_path = Path.unit_regular_polygon(4)
            marker_paths.append(marker_path)

        scatter.set_paths(marker_paths)

        ax.set_xlim(np.min(x_values) - padding, np.max(x_values) + padding)
        ax.set_ylim(np.min(y_values) - padding, np.max(y_values) + padding)

        # Plot scatter coordinates text
        for i, (x, y, sym, color) in enumerate(zip(x_values, y_values, symbol[: frame + 1], colors_values)):
            ax.text(x, y, str(sym), color=color, fontsize=text_size)

    animation = FuncAnimation(fig, update, frames=frames, blit=False)

    writer = FFMpegWriter(fps=fps)
    animation.save(output_path, writer=writer)






# Indexes and synthetic data creation

def create_random_climate_data(start,end,freq):
    dt_index = pd.date_range(start=start, end=end, freq=freq)

    # Create a dataframe with random temperature, humidity, and pressure data
    
    data = {'temperature': np.random.uniform(low=-60, high=60, size=len(dt_index)),
        'humidity': np.random.uniform(low=20, high=80, size=len(dt_index)),
        'pressure': np.random.uniform(low=980, high=1020, size=len(dt_index))}
        
    climate_data = pd.DataFrame(data=data, index=dt_index)
    
    return climate_data

def extract_climate_data(df):
    # Extract the datetime components as numpy arrays
    year = df.index.year.values
    month = df.index.month.values
    day = df.index.day.values
    hour = df.index.hour.values
    minute = df.index.minute.values
    second = df.index.second.values

    # Extract the data columns as numpy arrays
    temperature = df['temperature'].values
    humidity = df['humidity'].values
    pressure = df['pressure'].values
    
    return year, month, day, hour, minute, second, temperature, humidity, pressure

def month_range(year, month):
    if month == 2 and calendar.isleap(year):
        return 29
    return calendar.monthrange(year, month)[1]
