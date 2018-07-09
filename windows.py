import numpy as np

WINDOW_SIZE = 64
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
SCALES = [1.3, 1.7, 2.3, 2.8, 3.3]
XY_CELLS_PER_STEP = [(2,1), (2,4), (2,4), (2,6), (2,6)]
Y_START_STOP = [(390, 480), (385, 480), (390, 470), (380, 450), (380, 450)]

class Windows(object):
    def __init__(self, img_shape, scales=SCALES, y_start_stop=Y_START_STOP, 
                 xy_cells_per_step=XY_CELLS_PER_STEP, window_size=WINDOW_SIZE,
                 pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK):

        assert len(scales) == len(y_start_stop) == len(xy_cells_per_step), \
            "Length of parameter arrays must be equal"
        num_window_sets = len(scales)
        self.img_shape = img_shape
        self.xs_scales = scales
        self.y_start_stop = y_start_stop
        self.xy_cells_per_step = xy_cells_per_step
        self.window_size = window_size
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.windows = []

        for i in range(num_window_sets):
            self._set_windows(y_start_stop[i], scales[i], xy_cells_per_step[i])
    
    def _set_windows(self, y_start_stop, scale, xy_cells_per_step):

        ystart = y_start_stop[0]
        ystop  = y_start_stop[1]

        x_cells_per_step = xy_cells_per_step[0]
        y_cells_per_step = xy_cells_per_step[1]

        search_area_shape = (abs(ystop - ystart), self.img_shape[1], self.img_shape[2])

        nxblocks = (search_area_shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (search_area_shape[0] // self.pix_per_cell) - self.cell_per_block + 1 
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        nblocks_per_window = (self.window_size // self.pix_per_cell) - self.cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // x_cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // y_cells_per_step + 1
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                xpos = xb * x_cells_per_step
                ypos = yb * y_cells_per_step

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                win_draw = np.int(self.window_size * scale)
                xbox_left = self.img_shape[1] - np.int(xleft * scale) - win_draw  - 1
                ytop_draw = np.int(ytop * scale)
                
                ybox_left = ytop_draw + ystart
                xbox_right = xbox_left + win_draw
                ybox_right = ytop_draw + win_draw + ystart

                if xbox_left < 0 or xbox_right < 0:
                    continue

                if ybox_left >= self.img_shape[0]:
                    continue
                    
                self.windows.append(((xbox_left, ybox_left), (xbox_right, ybox_right)))

