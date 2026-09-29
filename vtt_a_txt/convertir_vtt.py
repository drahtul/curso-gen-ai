import os
import glob
import re

def vtt_a_txt(vtt_path):
    # Conserva el nombre original y cambia únicamente la extensión de salida.
    txt_path = os.path.splitext(vtt_path)[0] + '.txt'
    
    # Se ignoran errores de codificación para poder procesar subtítulos que
    # contengan algún carácter inválido sin detener toda la conversión.
    with open(vtt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lineas = f.readlines()
        
    lineas_limpias = []
    linea_anterior = ""
    
    # Estos patrones permiten distinguir la información técnica del subtítulo
    # (tiempos y etiquetas) del texto que se quiere conservar.
    patron_tiempo = re.compile(r'\d{2}:\d{2}[\d:.]*\s*-->\s*\d{2}:\d{2}[\d:.]*')
    patron_etiquetas = re.compile(r'<[^>]+>')
    
    for linea in lineas:
        # strip() elimina saltos de línea y espacios innecesarios antes de
        # aplicar los filtros y guardar el texto.
        linea_str = linea.strip()
        
        # Omitir cabecera WEBVTT, bloques especiales, líneas vacías y números
        # de secuencia, porque no forman parte del contenido hablado.
        if not linea_str or linea_str.startswith('WEBVTT') or linea_str.startswith('NOTE') or linea_str.startswith('STYLE'):
            continue
        if linea_str.isdigit():
            continue
        if patron_tiempo.search(linea_str):
            continue
            
        # Eliminar etiquetas de formato o de hablante, por ejemplo <v Juan>,
        # <i> o </i>, dejando solamente el texto visible.
        texto_limpio = patron_etiquetas.sub('', linea_str).strip()
        
        if not texto_limpio:
            continue
            
        # Los subtítulos automáticos suelen repetir una línea al actualizarse.
        # Solo se elimina la repetición si aparece inmediatamente después.
        if texto_limpio != linea_anterior:
            lineas_limpias.append(texto_limpio)
            linea_anterior = texto_limpio
            
    # Escribir todas las líneas limpias en el archivo de texto, conservando
    # cada fragmento del subtítulo en una línea independiente.
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas_limpias))
        
    print(f"✓ Convertido: {os.path.basename(vtt_path)} -> {os.path.basename(txt_path)}")

def convertir_carpeta(directorio='.'):
    # glob busca únicamente archivos .vtt directamente dentro de la carpeta;
    # no recorre subcarpetas.
    archivos_vtt = glob.glob(os.path.join(directorio, '*.vtt'))
    
    if not archivos_vtt:
        print("No se encontraron archivos .vtt en este directorio.")
        return
        
    print(f"Procesando {len(archivos_vtt)} archivo(s) .vtt...\n")
    for archivo in archivos_vtt:
        # Cada archivo se convierte de forma independiente y genera un .txt
        # con el mismo nombre base.
        vtt_a_txt(archivo)
    print("\n¡Proceso completado!")

if __name__ == '__main__':
    # Este bloque solo se ejecuta al lanzar este archivo directamente, no al
    # importarlo desde otro módulo. Convierte los .vtt de la carpeta actual.
    convertir_carpeta('.')
