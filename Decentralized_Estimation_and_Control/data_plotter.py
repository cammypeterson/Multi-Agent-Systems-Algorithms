# Takenfrom https://github.com/randybeard/controlbook_public
# Adapted to allow 2D plots

from matplotlib import get_backend
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Arrow
import numpy as np

plt.ion()  # enable interactive drawing

class dataPlotter:
    def __init__(self, n):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 1    # Number of subplot rows
        self.num_cols = 2    # Number of subplot columns
        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=False, figsize=(20,10))
        # move_figure(self.fig, 500, 500)
        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.f_star_history = [[],[],[],[],[]]  # reference goal vector (Mx, My, Mxx, Myy, Mxy)
        self.f_p_history = [[],[],[],[],[]]  # goal vector (Mx, My, Mxx, Myy, Mxy)
        self.agent_state_x_history = []  # agent x history
        self.agent_state_y_history = []  # agent y history
        for i in range(n):
            self.agent_state_x_history.append([])     # create vectors for each of the agents
            self.agent_state_y_history.append([])     # create vectors for each of the agents

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], xlabel='X (m)', ylabel='Y (m)', title='Multi-Agent Motion', grid_on=False, draw_heading=True))
        self.handle.append(myPlot(self.ax[1], xlabel='t(s)', ylabel='Goal Vector Value', legend=['M10', 'M01', 'M20', 'M02', 'M11'], title='Goal Vectors'))


    def update(self, t, f_star, f_p, states):
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        for i in range(len(self.f_star_history)):   
            self.f_star_history[i].append(f_star[i][0])    # Reference goal vector, f_star
            self.f_p_history[i].append(f_p[i][0])          # Estimated goal vector, f(p)
        
        for i in range(len(self.agent_state_x_history)):
            self.agent_state_x_history[i].append(states[i][0][0])  # Agent i x pos
            self.agent_state_y_history[i].append(states[i][1][0])  # Agent i y pos
        # update the plots with associated histories
        self.handle[0].update(self.agent_state_x_history, self.agent_state_y_history)
        self.handle[1].update([self.time_history], self.f_p_history) # [self.f_star_history, self.f_p_history])

    def write_data_file(self):
        with open('io_data.npy', 'wb') as f:
            np.save(f, self.time_history)
            np.save(f, self.theta_history)
            np.save(f, self.torque_history)


class myPlot:
    ''' 
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None,
                 grid_on=True,
                 draw_heading=False):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []
        self.circles = []
        self.arrows = []

        # Configure the axes
        self.ax.set_ylabel(ylabel, fontsize=16)
        self.ax.set_xlabel(xlabel, fontsize=16)
        self.ax.set_title(title, fontsize=20)
        self.ax.grid(grid_on)

        self.draw_heading = draw_heading

        # Keeps track of initialization
        self.init = True   

    def update(self, x_axis, data):
        ''' 
            Adds data to the plot.  
            x_axis is a list or a list of lists each corresponding to a line on the plot, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                if len(x_axis) < 2:
                    self.line.append(Line2D(x_axis,
                                            data[i],
                                            color=self.colors[np.mod(i, len(self.colors) - 1)],
                                            ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                            label=self.legend[i] if self.legend != None else None))
                else:
                    self.line.append(Line2D(x_axis[i],
                                            data[i],
                                            color=self.colors[np.mod(i, len(self.colors) - 1)],
                                            ls=self.line_styles[0],
                                            label=self.legend[i] if self.legend != None else None))
                
                # Initialize the circles and the arrows
                if self.draw_heading:
                    self.circles.append(Circle((x_axis[i][0], data[i][0]),
                                               0.3, fill=False))
                    self.arrows.append(Arrow(x_axis[i][0],
                                                 data[i][0],
                                                 0.0,
                                                 0.0, width=0.4, color='k'))
                    self.ax.add_patch(self.arrows[i])
                    self.ax.add_patch(self.circles[i])

                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line, fontsize=14)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                if len(x_axis) < 2:
                    self.line[i].set_xdata(x_axis)
                else:
                    self.line[i].set_xdata(x_axis[i])
                    self.ax.axis('square')
                self.line[i].set_ydata(data[i])
                if self.draw_heading:
                    dx1 = (x_axis[i][-1] - x_axis[i][-2])
                    dy1 = (data[i][-1] - data[i][-2]) 
                    # Normalize
                    dy = np.sign(dy1) * 1 / ( 1 + dx1**2 / dy1**2 )**0.5
                    dx = dx1 / dy1 * dy

                    ar = self.arrows[i]
                    ar.remove()
                    self.arrows[i] = Arrow(x_axis[i][-1],
                                           data[i][-1],
                                           dx,
                                           dy, width = 0.4, color='k')
                    self.ax.add_patch(self.arrows[i])

                    cir = self.circles[i]
                    cir.remove()
                    self.circles[i] = Circle((x_axis[i][-1],
                                             data[i][-1]),
                                             0.15, fill=False, lw=2.0)
                    self.ax.add_patch(self.circles[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
        plt.draw()

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    figmgr = plt.get_current_fig_manager()
    figmgr.canvas.manager.window.raise_()
    geom = figmgr.window.geometry()
    x,y,dx,dy = geom.getRect()
    figmgr.window.setGeometry(10, 10, dx, dy)
    # backend = get_backend()
    # if backend == 'TkAgg':
    #     f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    # elif backend == 'WXAgg':
    #     f.canvas.manager.window.SetPosition((x, y))
    # else:
    #     # This works for QT and GTK
    #     # You can also use window.setGeometry
    #     #f.canvas.manager.window.move(x, y)
    #     f.canvas.manager.setGeometry(x, y)

# f, ax = plt.subplots()
# move_figure(f, 500, 500)
# plt.show()