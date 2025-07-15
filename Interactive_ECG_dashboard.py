import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import wfdb
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import torch.nn as nn

# cd /Users/jiahaoshao/Documents/GitHub/Deep-Learning-of-ECG-signals

class ECGModel(nn.Module):
    def __init__(self, num_filters, kernel_size, dropout_rate):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters,   kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size=kernel_size)
        self.pool  = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size=kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc    = nn.Linear(num_filters*4, 1)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)        # (batch,1, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.global_avg_pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x

# Paths to the datasets for each class
class_directories = {
    'Class A': 'Data/Class_A',
    'Class N': 'Data/Class_N'
}

# Function to read ECG data from a single file
def read_ecg(file_path):
    record = wfdb.rdrecord(file_path)
    return record.p_signal, record.sig_name, record.fs  # Also return sampling frequency

# load the trained CNN
DEVICE = torch.device("cpu")
BEST_NUM_FILTERS = 104
BEST_KERNEL_SIZE = 4
BEST_DROPOUT_RATE = 0.2
model = ECGModel(BEST_NUM_FILTERS, BEST_KERNEL_SIZE, BEST_DROPOUT_RATE).to(DEVICE)

model.load_state_dict(torch.load("ecg_cnn.pt", map_location=DEVICE))
model.eval()

# ── Sub-window length from training script ──
SUB_TIMEWINDOW = 960

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('ECG Data Dashboard', style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label('Select ECG Class:'),
            dcc.Dropdown(
                id='class-dropdown',
                options=[{'label': k, 'value': v} for k, v in class_directories.items()],
                value='Data/Class_A',  # Default value
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label('Select ECG File:'),
            dcc.Dropdown(
                id='file-dropdown',
                options=[],
                value=None,
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'}),
        html.Div([
            html.Label('Select Plot Type:'),
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': 'ECG Signal Over Time', 'value': 'time'},
                    {'label': 'Histogram of Signal Amplitudes', 'value': 'histogram'},
                    {'label': 'Rolling Average', 'value': 'rolling'}
                ],
                value='time',
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '0 10px'})
    ], style={'display': 'flex'}),
    dcc.Graph(id='ecg-plot'),
    html.Div(id='classification-result', style={'marginTop':'20px','fontSize':'18px'})
])

@app.callback(
    Output('file-dropdown', 'options'),
    Input('class-dropdown', 'value')
)
def update_file_dropdown(class_path):
    return [
        {'label': name[:-4], 'value': os.path.join(class_path, name[:-4])}
        for name in os.listdir(class_path) if name.endswith('.hea')
    ]

@app.callback(
    [Output('ecg-plot', 'figure'),
     Output('classification-result', 'children')],
    [Input('file-dropdown', 'value'),
     Input('plot-type-dropdown', 'value')]
)

def update_output(selected_file, plot_type):
    if not selected_file:
        return go.Figure(), ""

    # 1) Read ECG data
    signals, labels, fs = read_ecg(selected_file)
    t_full = np.linspace(0, len(signals) / fs, num=len(signals))

    # 2) Build the plot (same as before)…
    fig = go.Figure()
    if plot_type == 'time':
        for i, lbl in enumerate(labels):
            fig.add_trace(go.Scatter(x=t_full, y=signals[:,i], name=lbl))
        fig.update_layout(title="ECG Signal Over Time", xaxis_title="Time (s)", yaxis_title="Amp")
    elif plot_type == 'histogram':
        for i, lbl in enumerate(labels):
            fig.add_trace(go.Histogram(x=signals[:,i], name=lbl, opacity=0.75))
        fig.update_layout(barmode='overlay', title="Histogram of Amplitudes")
    elif plot_type == 'rolling':
        w=50
        for i, lbl in enumerate(labels):
            avg = np.convolve(signals[:,i], np.ones(w)/w, mode='valid')
            fig.add_trace(go.Scatter(x=t_full[:len(avg)], y=avg, name=lbl))
        fig.update_layout(title="Rolling Average", xaxis_title="Time (s)", yaxis_title="Rolling Amp")

    # 3) Slide windows, normalize, and infer
    winsize = SUB_TIMEWINDOW  # 960
    stride  = winsize // 2    # 50% overlap
    probs = []
    for start in range(0, len(signals)-winsize+1, stride):
        win = signals[start:start+winsize, 0]  # lead 0
        # normalize
        win = (win - win.mean()) / (win.std() + 1e-6)
        inp = torch.tensor(win, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            p = model(inp.to(DEVICE)).item()
        probs.append(p)
        print(f"Window {start}-{start+winsize}: p={p:.3f}")

    # sometimes tail is left—optionally include last segment:
    if len(signals) % stride != 0:
        tail = signals[-winsize:,0]
        tail = (tail - tail.mean()) / (tail.std() + 1e-6)
        inp = torch.tensor(tail, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            p = model(inp.to(DEVICE)).item()
        probs.append(p)
        # print(f"Tail window: p={p:.3f}")

    max_p = float(np.max(probs))
    # print(f"Max probability over all windows: {max_p:.3f}")

    label = "Abnormal" if max_p > 0.5 else "Normal"
    result_text = f"Classification: {label} (max p={max_p:.2f})"

    return fig, result_text

if __name__ == '__main__':
#    app.run_server(debug=True)
#     # app.run(port=8050, debug=True)
    app.run_server(mode='external', host='0.0.0.0', port=8050, debug=True)


