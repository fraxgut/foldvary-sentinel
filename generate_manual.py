from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Centinela F2628 - Manual Operativo de Señales', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Página ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, label, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, logic, description):
        self.set_font('Courier', '', 10)
        self.set_text_color(100, 0, 0)
        self.multi_cell(0, 5, f"Lógica: {logic}")
        self.ln(2)
        
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, f"Descripción: {description}")
        self.ln(8)

pdf = PDF()
pdf.alias_nb_pages()
pdf.add_page()

# Intro
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 5, "Este documento detalla la lógica operativa del Centinela F2628 (v2026). Las señales han sido actualizadas para utilizar umbrales estadísticos adaptativos (Z-Score, Percentiles, Bandas de Bollinger) en lugar de valores fijos arbitrarios.")
pdf.ln(10)

# SECTION 1: CRITICAL SIGNALS
pdf.chapter_title("1. SEÑALES CRÍTICAS (ROJO)")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "COMBO_CRISIS (La Tormenta Perfecta)", 0, 1)
pdf.chapter_body(
    "Spread > 5.0% AND Spread_3d_Confirm AND SPX >= High50 AND RSI < High50_RSI AND VIX < 13",
    "La señal más peligrosa. Ocurre cuando el mercado de crédito se congela (nadie presta, Spread > 5%) pero la bolsa sigue en máximos históricos con complacencia extrema (VIX bajo). Es la divergencia final antes de un crash de liquidez."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "SOLVENCY_DEATH (Crisis de Solvencia)", 0, 1)
pdf.chapter_body(
    "Spread HY > 5.0% (confirmado por 3 días consecutivos)",
    "El diferencial (spread) entre bonos basura y bonos del tesoro supera el 5%. Indica que los inversores exigen una prima de riesgo masiva porque temen una ola de quiebras corporativas. El nivel del 5% es el estándar de la industria para definir estrés crediticio sistémico."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "TEMPORAL_CRISIS (Convergencia Temporal)", 0, 1)
pdf.chapter_body(
    "(SUGAR_CRASH + EM_CURRENCY_STRESS en <30 días) OR (SOLVENCY_DEATH + WAR_PROTOCOL en <30 días)",
    "No es un evento único, sino una secuencia. Detecta cuando múltiples fracturas estructurales ocurren en un periodo corto, indicando que el sistema no puede contener las fugas por más tiempo."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "HOUSING_BUST (Giro del Ciclo Foldvary)", 0, 1)
pdf.chapter_body(
    "Housing_Starts_ZScore < -1.5 AND Mortgage_Rate > 52_Week_Avg",
    "El motor de la economía (construcción) se frena bruscamente (1.5 desviaciones estándar por debajo de la media) MIENTRAS las tasas hipotecarias están en tendencia alcista (superando su promedio anual). Esto confirma debilidad fundamental en el ciclo de tierras."
)

# SECTION 2: WARNING SIGNALS
pdf.chapter_title("2. SEÑALES DE ALERTA (AMARILLO/NARANJA)")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "BOND_FREEZE (Congelamiento de Bonos)", 0, 1)
pdf.chapter_body(
    "US10Y > Z_Score(Mean + 2*StdDev) AND RSI > 70",
    "Venta de pánico en el mercado de bonos del Tesoro. Los rendimientos se disparan estadísticamente fuera de lo normal (2 desviaciones estándar), encareciendo toda la deuda global."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "EM_CURRENCY_STRESS (Estrés Emergente)", 0, 1)
pdf.chapter_body(
    "DXY > Percentil_95 (1 año) AND US10Y > SMA_250",
    "El dólar alcanza niveles estadísticamente extremos (top 5% del último año) Y los bonos del Tesoro (US10Y) están en tendencia alcista (sobre su media móvil de 250 días). Esta combinación drena liquidez global y presiona la deuda de mercados emergentes."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "WAR_PROTOCOL (Protocolo de Guerra)", 0, 1)
pdf.chapter_body(
    "Oil > Z_Score(Mean + 2*StdDev) AND Gold > High20 AND SPX < Low20",
    "Shock geopolítico. El petróleo se dispara por miedo a cortes de suministro, el oro sube como refugio, y las acciones caen. El mercado está descontando conflicto bélico."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "SUGAR_CRASH (Euforia Terminal)", 0, 1)
pdf.chapter_body(
    "SPX >= High50 AND RSI < High50_RSI (Divergencia) AND VIX < 13",
    "Una trampa alcista. El precio hace un nuevo máximo, pero la fuerza (RSI) es menor que antes, y nadie compra protección (VIX bajo). El mercado está 'borracho' y vulnerable."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "LABOUR_SHOCK (Shock Laboral)", 0, 1)
pdf.chapter_body(
    "ICSA > Z_Score(Mean + 2*StdDev)",
    "Ruptura estadística en el empleo. Las solicitudes de desempleo se desvían más de 2 desviaciones estándar de su media anual. Indica un cambio de régimen repentino en el mercado laboral."
)

# SECTION 3: TRADING SIGNALS
pdf.chapter_title("3. SEÑALES DE TRADING (VERDE/SALIDA)")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "BUY_BTC_NOW (Entrada Bitcoin)", 0, 1)
pdf.chapter_body(
    "Net_Liquidity > SMA(10) AND Prev_Net_Liq < SMA_Prev AND RSI < 60",
    "La Reserva Federal abre el grifo. La Liquidez Neta cruza al alza su media móvil, indicando expansión monetaria. Históricamente, el mejor momento para comprar BTC."
)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, "BUY_WPM_NOW (Entrada WPM)", 0, 1)
pdf.chapter_body(
    "Price < Bollinger_Lower_Band (Mean - 2*StdDev) AND RSI < 30 AND Volume > 2 * Avg_Vol",
    "Capitulación estadística. El precio rompe la banda de Bollinger inferior (2 sigmas), confirmando que está barato con un 95% de confianza estadística, acompañado de pánico vendedor (volumen)."
)

pdf.output("es-foldvarysignalmanual.pdf")
print("PDF generated successfully.")