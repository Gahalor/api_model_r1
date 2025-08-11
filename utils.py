import json
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import iirnotch, filtfilt, coherence
from scipy.interpolate import interp1d

# ========= CONFIGURACIÓN ===========
# Parámetros de configuración
electrodes_distance_in_meters = 2.0

# Recorte
buffer_in_seconds = 0

# Definir permeabilidad relativa del material (adimensional, por defecto 1)
mur = 1.0 

# Definir tipo de preprocesamiento
demean = True
taper = True
notch = True
notch_freq = 50

# definir ventana de análisis (largo de serie de tiempo)
max_time_msec = 200 

def safe_get(data, *keys, default=None):
    """Acceso seguro a claves anidadas."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

def read_json_file(file_path, buffer_in_seconds=0.0):
    """Lee y procesa un archivo JSON de datos sísmicos y eléctricos."""
    with open(file_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Soporte para 'data' como root si aplica
    data = json_data.get("data", json_data)

    # Obtener fuentes de datos según estructura
    seismoelectric_data = safe_get(data, "seismoelectric", "data") or safe_get(data, "samples", "seismoelectric", "data") or {}
    magnetometer_data = safe_get(data, "magnetometer", "data") or safe_get(data, "samples", "magnetometer", "data") or {}
    
    if safe_get(data, "samples", "magnetometer", "sampleRate") is not None:
        samplerate = safe_get(data, "samples", "magnetometer", "sampleRate")
    else:
        samplerate = safe_get(data, "magnetometer", "samplerate")

    # Convertir datos a arrays numpy
    elec_v1 = np.array(seismoelectric_data.get("v1", []), dtype=float) * 1e-3 # in V
    elec_v2 = np.array(seismoelectric_data.get("v2", []), dtype=float) * 1e-3

    magn_x = np.array(magnetometer_data.get("x", []), dtype=float) * 1e-9 # in T
    magn_y = np.array(magnetometer_data.get("y", []), dtype=float) * 1e-9
    magn_z = np.array(magnetometer_data.get("z", []), dtype=float) * 1e-9

    # Vector de tiempo (si no hay samplerate, asumir 3333 Hz por defecto para evitar error)
    sampling = samplerate if samplerate else 3333
    dt = 1 / sampling
    time_s = np.arange(0, len(magn_x) * dt, dt)

    # Construcción del DataFrame
    df = pd.DataFrame(np.array([
        time_s, elec_v1, elec_v2, magn_x, magn_y, magn_z
    ]).T, columns=['time','V1','V2','Bx','By','Bz'])

    # Recorte por buffer si se indica
    if buffer_in_seconds > 0:
        df = df[df['time'] >= buffer_in_seconds].copy()
        df['time'] = df['time'] - df['time'].iloc[0]
        df = df.reset_index(drop=True)

    # Metadata segura
    metadata = {
        "projectName": data.get("projectName", "undefined"),
        "timezone": data.get("timezone", "undefined"),
        "timestamp": data.get("timestamp", "undefined"),
        "geolocation": data.get("geolocation", []),
        "deviceId": data.get("deviceId", "undefined"),
        "temperature": data.get("temperature", None),
        "humidity": data.get("humidity", None),
        "sampling": sampling
    }

    return df, metadata

def waveform_preprocessing(array, fs, demean=True, notch=False, notch_freq=50, quality_factor=30):
    """Preprocesa señales: remueve promedio y aplica filtro notch."""
    # Remover promedio
    if demean:
        array = array - np.mean(array)
    # Aplicar filtro notch para atenuar una banda estrecha
    if notch:
        w0 = notch_freq / (fs / 2)
        b, a = iirnotch(w0, quality_factor)
        array = filtfilt(b, a, array)    
    return array

def convert_B_to_H(B, mur=1, mu0=4*np.pi*1e-7):
    """Convierte campo magnético B a campo H."""
    # Se calcula H dividiendo el flujo magnético B por el producto de mu0 y mur.
    H = B / (mu0*mur)
    return H

def convert_B_to_Hm(Bm, mur=1, mu0=4*np.pi*1e-7):
    # Se calcula H dividiendo el flujo magnético B por el producto de mu0 y mur.
    Hm = Bm / (mu0*mur) 
    return Hm

def compute_spectral_analysis(E_signal, H_signal, fs, nperseg=256, noverlap=None):
    """
    Calcula el análisis espectral de las señales E y H.
    
    Parámetros:
      E_signal, H_signal:
         Vectores de señal (procesados: sin tendencia y con taper).
      fs: frecuencia de muestreo (Hz).
    Devuelve:
      E_amp_vec: Amplitud espectral de la señal eléctrica.
      H_amp_vec: Amplitud espectral de la señal magnética.
      E_pha_vec: Fase de la señal eléctrica (grados).
      H_pha_vec: Fase de la señal magnética (grados).
      EH_coh_vec: Coherencia entre E_signal y H_signal.
      f: Vector de frecuencias (Hz).
    """
    # Utiliza FFT para obtener el espectro completo
    dt = 1/fs
    n = len(E_signal)
    E_fft = fft(E_signal)
    H_fft = fft(H_signal)
    freqs = fftfreq(n, dt)
    pos_mask = freqs > 0
    f = freqs[pos_mask]
    E_amp_vec = (2.0 / n) * abs(np.sqrt(E_fft[pos_mask] * np.conjugate(E_fft[pos_mask]))) # np.abs(E_fft[pos_mask])
    H_amp_vec = (2.0 / n) * abs(np.sqrt(H_fft[pos_mask] * np.conjugate(H_fft[pos_mask]))) # np.abs(H_fft[pos_mask])
    E_pha_vec = np.angle(E_fft[pos_mask], deg=True)
    H_pha_vec = np.angle(H_fft[pos_mask], deg=True)

    # Calcular la coherencia entre E_signal y H_signal
    if noverlap is None:
        noverlap = nperseg // 2

    f_coh, EH_coh_vec = coherence(E_signal, H_signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    # Si es necesario, interpola la coherencia para que tenga el mismo vector de frecuencias que 'f'
    if len(f_coh) != len(f):
        EH_coh_vec = np.interp(f, f_coh, EH_coh_vec)
    
    return E_amp_vec, H_amp_vec, E_pha_vec, H_pha_vec, EH_coh_vec, f

def calculate_resistivity(E_amp_vec,H_amp_vec, mur=1.0, mu0=4*np.pi*1e-7):
    """Calcula la resistividad a partir de las amplitudes espectrales."""
    # Evitar división por cero
    H_amp_vec = np.where(H_amp_vec == 0, 1e-12, H_amp_vec)
    # Calcular la impedancia espectral
    Z_vec = E_amp_vec / H_amp_vec
    # Calcular la resistividad espectral: ρ = |Z|² / μ₀
    rho_vec = (Z_vec**2) / (mu0*mur)
    return rho_vec

def calculate_skin_depth(rho, f_vec, mu0=4*np.pi*1e-7, zmin=0, zmax=300):
    """
    Calcula la profundidad de penetración (skin depth) y 
    'recorta' los resultados en el rango [zmin, zmax].

    Parámetros:
    -----------
    rho : float or array
        Resistividad [Ohm·m].
    f_vec : float or array
        Frecuencia o vector de frecuencias [Hz].
    mu0 : float
        Permeabilidad del vacío (default = 4*pi*1e-7 H/m).
    zmin, zmax : float
        Límites inferior y superior para recortar delta.
    
    Retorna:
    --------
    delta_recortado : array
        Profundidades de penetración recortadas en [zmin, zmax].
    rho_recortado : array
        Resistividades correspondientes a los valores de delta recortados.
    """
    # 1) Calculamos la frecuencia angular
    w = 2*np.pi*f_vec

    # 2) Calculamos la profundidad de penetración (skin depth)
    delta = np.sqrt((2*rho)/(w*mu0))

    # 3) Ordenamos por valor de delta (creciente)
    args = np.argsort(delta)
    delta = delta[args]
    rho = rho[args]

    # 4) Creamos una máscara para recortar valores de delta en [zmin, zmax]
    mask = (delta >= zmin) & (delta <= zmax)

    # 5) Aplicamos el corte en delta y rho
    delta_recortado = delta[mask]
    rho_recortado = rho[mask]

    # 6) Retornamos los valores recortados
    return delta_recortado, rho_recortado

def resitivity_depth_interpolation(x, y, num_points=10, kind='nearest', space='log'):
    """
    Interpola (x, y) en una malla de x logarítmica.

    Parámetros:
    -----------
    x : array-like
        Valores originales de x (deben ser > 0).
    y : array-like
        Valores correspondientes de y.
    num_points : int
        Cuántos puntos crear en la malla logarítmica.
    kind : str
        Specifies the kind of interpolation as a string or as an integer specifying the order of 
        the spline interpolator to use. The string has to be one of 'linear', 'nearest', 
        'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 
        'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of zeroth, 
        first, second or third order; 'previous' and 'next' simply return the previous or 
        next value of the point; 'nearest-up' and 'nearest' differ when interpolating 
        half-integers (e.g. 0.5, 1.5) in that 'nearest-up' rounds up and 'nearest' rounds down. 
        Default is 'linear'.

    Retorna:
    --------
    x_new : np.array
        Vector de x espaciado logarítmicamente.
    y_new : np.array
        Valores de y interpolados en x_new.
    """

    # Asegurar arrays NumPy y ordenarlos por x
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    # Ordenar de menor a mayor
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Extraer mínimos y máximos para logspace
    x_min, x_max = x_sorted[0], x_sorted[-1]

    # Verificar que x_min > 0 para poder usar log10
    if x_min <= 0:
        raise ValueError("Todos los valores de x deben ser positivos para usar logspace.")

    # Crear la malla logarítmica
    if space=='log':
        x_new = np.logspace(np.log10(x_min), np.log10(x_max), num=num_points)
    if space=='lin':
        x_new = np.linspace(x_min, x_max, num=num_points)

    # Definir la función de interpolación
    f = interp1d(x_sorted, y_sorted, kind=kind, fill_value='extrapolate')

    # Interpolar en la malla log
    y_new = f(x_new)
    
    return x_new, y_new

def process_json_data(json_data):
    """
    Función que procesa directamente un diccionario JSON sin necesidad de archivo.
    
    Parámetros:
    -----------
    json_data : dict
        Diccionario con los datos JSON ya parseados.
        
    Retorna:
    --------
    dict : Diccionario con todos los datos procesados organizados por gráfico.
    """
    # Soporte para 'data' como root si aplica
    data = json_data.get("data", json_data)

    # Obtener fuentes de datos según estructura
    seismoelectric_data = safe_get(data, "seismoelectric", "data") or safe_get(data, "samples", "seismoelectric", "data") or {}
    magnetometer_data = safe_get(data, "magnetometer", "data") or safe_get(data, "samples", "magnetometer", "data") or {}
    
    if safe_get(data, "samples", "magnetometer", "sampleRate") is not None:
        samplerate = safe_get(data, "samples", "magnetometer", "sampleRate")
    else:
        samplerate = safe_get(data, "magnetometer", "samplerate")

    # Convertir datos a arrays numpy
    elec_v1 = np.array(seismoelectric_data.get("v1", []), dtype=float) * 1e-3 # in V
    elec_v2 = np.array(seismoelectric_data.get("v2", []), dtype=float) * 1e-3

    magn_x = np.array(magnetometer_data.get("x", []), dtype=float) * 1e-9 # in T
    magn_y = np.array(magnetometer_data.get("y", []), dtype=float) * 1e-9
    magn_z = np.array(magnetometer_data.get("z", []), dtype=float) * 1e-9

    # Vector de tiempo (si no hay samplerate, asumir 3333 Hz por defecto para evitar error)
    sampling = samplerate if samplerate else 3333
    dt = 1 / sampling
    time_s = np.arange(0, len(magn_x) * dt, dt)

    # Construcción del DataFrame
    df = pd.DataFrame(np.array([
        time_s, elec_v1, elec_v2, magn_x, magn_y, magn_z
    ]).T, columns=['time','V1','V2','Bx','By','Bz'])

    # Recorte por buffer si se indica
    if buffer_in_seconds > 0:
        df = df[df['time'] >= buffer_in_seconds].copy()
        df['time'] = df['time'] - df['time'].iloc[0]
        df = df.reset_index(drop=True)

    # Metadata segura
    metadata = {
        "sampling": sampling
    }

    # Usar la misma lógica que process_resistivity_data pero con los datos ya extraídos
    return _process_dataframe_and_metadata(df, metadata)

def _process_dataframe_and_metadata(df, metadata):
    """
    Función auxiliar que procesa el DataFrame y metadata para generar todos los datos.
    Esta función contiene la lógica común entre process_resistivity_data y process_json_data.
    """
    # Frecuencia de muestreo y vector de tiempo
    fs = metadata['sampling']
    timevec = df['time'].values

    # Campos eléctricos
    V1 = df['V1'].values 
    V2 = df['V2'].values 

    # Campos magnéticos
    Bx = df['Bx'].values 
    By = df['By'].values 
    Bz = df['Bz'].values 

    # Calcular el módulo del campo magnético
    Bm = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Preprocesamiento: demean + taper
    V1 = waveform_preprocessing(V1, fs, demean, taper, notch, notch_freq)
    V2 = waveform_preprocessing(V2, fs, demean, taper, notch, notch_freq)
    Bx = waveform_preprocessing(Bx, fs, demean, taper, notch, notch_freq)
    By = waveform_preprocessing(By, fs, demean, taper, notch, notch_freq)
    Bz = waveform_preprocessing(Bz, fs, demean, taper, notch, notch_freq)

    # Calcular campo eléctrico y magnético
    E1 = V1 / electrodes_distance_in_meters    
    E2 = V2 / electrodes_distance_in_meters    

    Hx = convert_B_to_H(Bx,mur)
    Hy = convert_B_to_H(By,mur)
    Hz = convert_B_to_H(Bz,mur)

    Hm = convert_B_to_Hm(Bm,mur)
    
    # Análisis espectral, cálculo de resistividad y skin depth
    inds = np.where(timevec <= max_time_msec/1000)[0]
    timevec = timevec[inds]
    V1 = V1[inds]
    V2 = V2[inds]
    E1 = E1[inds]
    E2 = E2[inds]
    Hx = Hx[inds]
    Hy = Hy[inds]
    Hz = Hz[inds]
    Hm = Hm[inds]

    # E1 - Hx
    E1_amp, Hx_amp, E1_pha, Hx_pha, E1Hx_coh, f_vec = compute_spectral_analysis(E1, Hx, fs)
    rho_xx = calculate_resistivity(E1_amp, Hx_amp, mur)
    inds = np.where(E1Hx_coh>0.5)[0]
    delta_xx_1, rho_xx_1 = calculate_skin_depth(rho_xx[inds], f_vec[inds])
    rho_xx_1 = rho_xx_1 * 2.46  

    # E1 - Hy
    E1_amp, Hy_amp, E1_pha, Hy_pha, E1Hy_coh, f_vec = compute_spectral_analysis(E1, Hy, fs)
    rho_xy = calculate_resistivity(E1_amp, Hy_amp, mur)
    inds = np.where(E1Hy_coh>0.5)[0]
    delta_xy_1, rho_xy_1 = calculate_skin_depth(rho_xy[inds], f_vec[inds])
    rho_xy_1 = rho_xy_1 * 2.46

    # E1 - Hz
    E1_amp, Hz_amp, E1_pha, Hz_pha, E1Hz_coh, f_vec = compute_spectral_analysis(E1, Hz, fs)
    rho_xz = calculate_resistivity(E1_amp, Hz_amp, mur)
    inds = np.where(E1Hz_coh>0.5)[0]
    delta_xz_1, rho_xz_1 = calculate_skin_depth(rho_xz[inds], f_vec[inds])
    rho_xz_1 = rho_xz_1 * 2.46

    # E2 - Hx
    E2_amp, Hx_amp, E2_pha, Hx_pha, E2Hx_coh, f_vec = compute_spectral_analysis(E2, Hx, fs)
    rho_xx = calculate_resistivity(E2_amp, Hx_amp, mur)
    inds = np.where(E2Hx_coh>0.5)[0]
    delta_xx_2, rho_xx_2 = calculate_skin_depth(rho_xx[inds], f_vec[inds])
    rho_xx_2 = rho_xx_2 * 2.46

    # E2 - Hy
    E2_amp, Hy_amp, E2_pha, Hy_pha, E2Hy_coh, f_vec = compute_spectral_analysis(E2, Hy, fs)
    rho_xy = calculate_resistivity(E2_amp, Hy_amp, mur)
    inds = np.where(E2Hy_coh>0.5)[0]
    delta_xy_2, rho_xy_2 = calculate_skin_depth(rho_xy[inds], f_vec[inds])
    rho_xy_2 = rho_xy_2 * 2.46

    # E2 - Hz
    E2_amp, Hz_amp, E2_pha, Hz_pha, E2Hz_coh, f_vec = compute_spectral_analysis(E2, Hz, fs)
    rho_xz = calculate_resistivity(E2_amp, Hz_amp, mur)
    inds = np.where(E2Hz_coh>0.5)[0]
    delta_xz_2, rho_xz_2 = calculate_skin_depth(rho_xz[inds], f_vec[inds])
    rho_xz_2 = rho_xz_2 * 2.46
    

    # Generar datos interpolados para cada resistividad
    # Resistividad E1-Hx interpolada
    if len(rho_xx_1) > 0:
        depth_new_e1hx, rho_new_e1hx = resitivity_depth_interpolation(delta_xx_1, rho_xx_1)
    else:
        depth_new_e1hx, rho_new_e1hx = [], []
    # Resistividad E1-Hy interpolada
    if len(rho_xy_1) > 0:
        depth_new_e1hy, rho_new_e1hy = resitivity_depth_interpolation(delta_xy_1, rho_xy_1)
    else:
        depth_new_e1hy, rho_new_e1hy = [], []
    # Resistividad E1-Hz interpolada
    if len(rho_xz_1) > 0:
        depth_new_e1hz, rho_new_e1hz = resitivity_depth_interpolation(delta_xz_1, rho_xz_1)
    else:
        depth_new_e1hz, rho_new_e1hz = [], []

    # Resistividad E2-Hx interpolada
    if len(rho_xx_2) > 0:
        depth_new_e2hx, rho_new_e2hx = resitivity_depth_interpolation(delta_xx_2, rho_xx_2)
    else:
        depth_new_e2hx, rho_new_e2hx = [], []
    # Resistividad E2-Hy interpolada
    if len(rho_xy_2) > 0:
        depth_new_e2hy, rho_new_e2hy = resitivity_depth_interpolation(delta_xy_2, rho_xy_2)
    else:
        depth_new_e2hy, rho_new_e2hy = [], []
    # Resistividad E2-Hz interpolada
    if len(rho_xz_2) > 0:
        depth_new_e2hz, rho_new_e2hz = resitivity_depth_interpolation(delta_xz_2, rho_xz_2)
    else:
        depth_new_e2hz, rho_new_e2hz = [], []


    # Organizar todos los datos en un diccionario
    all_plot_data = {
        "resistivity_e1hx": {
            "rho_xx_1": rho_xx_1.tolist() if len(rho_xx_1) > 0 else [],
            "delta_xx_1": delta_xx_1.tolist() if len(delta_xx_1) > 0 else [],
            "rho_new": rho_new_e1hx.tolist() if len(rho_new_e1hx) > 0 else [],
            "depth_new": depth_new_e1hx.tolist() if len(depth_new_e1hx) > 0 else []
        },
        "resistivity_e1hy": {
            "rho_xy_1": rho_xy_1.tolist() if len(rho_xy_1) > 0 else [],
            "delta_xy_1": delta_xy_1.tolist() if len(delta_xy_1) > 0 else [],
            "rho_new": rho_new_e1hy.tolist() if len(rho_new_e1hy) > 0 else [],
            "depth_new": depth_new_e1hy.tolist() if len(depth_new_e1hy) > 0 else []
        },
        "resistivity_e1hz": {
            "rho_xz_1": rho_xz_1.tolist() if len(rho_xz_1) > 0 else [],
            "delta_xz_1": delta_xz_1.tolist() if len(delta_xz_1) > 0 else [],
            "rho_new": rho_new_e1hz.tolist() if len(rho_new_e1hz) > 0 else [],
            "depth_new": depth_new_e1hz.tolist() if len(depth_new_e1hz) > 0 else []
        },
        "resistivity_e2hx": {
            "rho_xx_2": rho_xx_2.tolist() if len(rho_xx_2) > 0 else [],
            "delta_xx_2": delta_xx_2.tolist() if len(delta_xx_2) > 0 else [],
            "rho_new": rho_new_e2hx.tolist() if len(rho_new_e2hx) > 0 else [],
            "depth_new": depth_new_e2hx.tolist() if len(depth_new_e2hx) > 0 else []
        },
        "resistivity_e2hy": {
            "rho_xy_2": rho_xy_2.tolist() if len(rho_xy_2) > 0 else [],
            "delta_xy_2": delta_xy_2.tolist() if len(delta_xy_2) > 0 else [],
            "rho_new": rho_new_e2hy.tolist() if len(rho_new_e2hy) > 0 else [],
            "depth_new": depth_new_e2hy.tolist() if len(depth_new_e2hy) > 0 else []
        },
        "resistivity_e2hz": {
            "rho_xz_2": rho_xz_2.tolist() if len(rho_xz_2) > 0 else [],
            "delta_xz_2": delta_xz_2.tolist() if len(delta_xz_2) > 0 else [],
            "rho_new": rho_new_e2hz.tolist() if len(rho_new_e2hz) > 0 else [],
            "depth_new": depth_new_e2hz.tolist() if len(depth_new_e2hz) > 0 else []
        },
    }

    return all_plot_data

def process_resistivity_data(filepath):
    """
    Función principal que procesa un archivo JSON y devuelve todos los datos para gráficos.
    
    Parámetros:
    -----------
    filepath : str
        Ruta al archivo JSON a procesar.
        
    Retorna:
    --------
    dict : Diccionario con todos los datos procesados organizados por gráfico.
    """
    df, metadata = read_json_file(filepath, buffer_in_seconds)
    return _process_dataframe_and_metadata(df, metadata)
