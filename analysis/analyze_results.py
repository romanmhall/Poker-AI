import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LEAGUE_LOG = os.path.join(DATA_DIR, "league_history.csv")
TRAIN_LOG = os.path.join(DATA_DIR, "training_history.csv")
OUTPUT_IMG = os.path.join(DATA_DIR, "progress_report.png")

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def analyze_training():
    print_header("TRAINING METRICS (The Brain)")
    if not os.path.exists(TRAIN_LOG):
        print("[WARN] No training history found yet.")
        return None

    try:
        df = pd.read_csv(TRAIN_LOG)
        if df.empty:
            print("[WARN] Training log is empty.")
            return None

        # Force conversion to numbers (coerce errors to NaN)
        df['Final_Loss'] = pd.to_numeric(df['Final_Loss'], errors='coerce')
        
        # Drop rows where Loss is missing
        df = df.dropna(subset=['Final_Loss'])

        if len(df) == 0:
            print("[WARN] Training log exists but contains no valid numbers.")
            return None

        total_runs = len(df)
        first_loss = df['Final_Loss'].iloc[0]
        last_loss = df['Final_Loss'].iloc[-1]
        
        print(f"Models Trained:    {total_runs}")
        print(f"Starting Loss:     {first_loss:.4f}")
        print(f"Current Loss:      {last_loss:.4f}")
        
        return df
    except Exception as e:
        print(f"[ERR] Could not read training log: {e}")
        return None

def analyze_league():
    print_header("LEAGUE PERFORMANCE (The Body)")
    if not os.path.exists(LEAGUE_LOG):
        print("[WARN] No league history found yet.")
        return None
        
    try:
        df = pd.read_csv(LEAGUE_LOG)
        if df.empty:
            print("[WARN] League log is empty.")
            return None

        # --- 1. SAFE DATA CONVERSION ---
        # Convert columns to numbers, forcing errors to NaN
        df['Win_Rate_BB100'] = pd.to_numeric(df['Win_Rate_BB100'], errors='coerce')
        df['VPIP'] = pd.to_numeric(df['VPIP'], errors='coerce') * 100
        df['PFR_Ratio'] = pd.to_numeric(df['PFR_Ratio'], errors='coerce')
        df['Agg_Score'] = pd.to_numeric(df['Agg_Score'], errors='coerce')
        
        # Handle 3-Bet (New vs Old)
        if 'ThreeBet_Pct' in df.columns:
            df['ThreeBet_Pct'] = pd.to_numeric(df['ThreeBet_Pct'], errors='coerce') * 100
        else:
            df['ThreeBet_Pct'] = 0.0

        # Handle Score (CPI vs Old Audit)
        # We try to grab 'CPI_Score' first. If missing, we look for 'Audit_Status'
        if 'CPI_Score' in df.columns:
            df['Score_Display'] = pd.to_numeric(df['CPI_Score'], errors='coerce')
        elif 'Audit_Status' in df.columns:
             # Legacy text support would go here, but for graphing we just set to 0
             df['Score_Display'] = 0.0
        else:
            df['Score_Display'] = 0.0

        # Drop rows with broken Win Rates for the Top 5 calculation
        valid_df = df.dropna(subset=['Win_Rate_BB100'])

        # --- 2. HALL OF FAME (Top 5) ---
        print("\n[HALL OF FAME] The Top 5 Models by Win Rate:")
        print(f"{'Model Name':<40} | {'Win Rate':<8} | {'VPIP':<5} | {'PFR':<5} | {'3-Bet':<5}")
        print("-" * 85)
        
        if not valid_df.empty:
            top_5 = valid_df.nlargest(5, 'Win_Rate_BB100')
            for _, row in top_5.iterrows():
                # Safe String Formatting
                winner_name = str(row['Winner'])
                name = (winner_name[:37] + '..') if len(winner_name) > 37 else winner_name
                
                print(f"{name:<40} | {row['Win_Rate_BB100']:<8.2f} | {row['VPIP']:<5.1f} | {row['PFR_Ratio']:<5.2f} | {row['ThreeBet_Pct']:<5.1f}%")
        else:
            print("   (No valid win rate data found yet)")

        # --- 3. CURRENT CHAMPION ---
        if not df.empty:
            current = df.iloc[-1]
            print("\n[CURRENT CHAMPION]")
            print(f"Name:        {current['Winner']}")
            # Use safe 'get' to avoid crashes if column missing
            wr = current.get('Win_Rate_BB100', 0.0)
            tb = current.get('ThreeBet_Pct', 0.0)
            print(f"Win Rate:    {float(wr):.2f} BB/100")
            print(f"3-Bet %:     {float(tb):.1f}% (Target: 6-15%)")
            
            # Show CPI if available, else Audit
            if 'CPI_Score' in current:
                print(f"CPI Score:   {float(current['CPI_Score']):.4f} / 1.0")
            elif 'Audit_Status' in current:
                print(f"Status:      {current['Audit_Status']}")

        return df

    except Exception as e:
        print(f"[ERR] Could not analyze league data: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_charts(train_df, league_df):
    if train_df is None and league_df is None: return

    plt.figure(figsize=(14, 12))
    plt.suptitle("Poker AI Evolution Report (Fixed)", fontsize=16)

    # 1. Neural Network Loss 
    # [Image of Loss Curve Graph]

    plt.subplot(3, 2, 1)
    if train_df is not None and not train_df.empty:
        plt.plot(train_df.index, train_df['Final_Loss'], label='Raw Loss', color='red', alpha=0.3)
        if len(train_df) > 5:
            plt.plot(train_df['Final_Loss'].rolling(window=5).mean(), label='Trend (5-Avg)', color='darkred', linewidth=2)
        plt.title("Brain Training Loss (Lower is Better)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No Training Data", ha='center')

    # 2. Win Rate
    plt.subplot(3, 2, 2)
    if league_df is not None and not league_df.empty:
        # Filter out NaNs for plotting
        clean_league = league_df.dropna(subset=['Win_Rate_BB100'])
        plt.plot(clean_league.index, clean_league['Win_Rate_BB100'], label='Win Rate', color='green', alpha=0.4)
        
        if len(clean_league) > 3:
            plt.plot(clean_league['Win_Rate_BB100'].rolling(window=3).mean(), label='Trend', color='darkgreen', linewidth=2)
            
        # Highlight Top 5
        if not clean_league.empty:
            top_5 = clean_league.nlargest(5, 'Win_Rate_BB100')
            plt.scatter(top_5.index, top_5['Win_Rate_BB100'], color='gold', s=100, edgecolors='black', zorder=5)
            
        plt.title("Champion Win Rate (BB/100)")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No League Data", ha='center')

    # 3. Aggression (PFR)
    plt.subplot(3, 2, 3)
    if league_df is not None and not league_df.empty:
        plt.plot(league_df.index, league_df['PFR_Ratio'], label='PFR Ratio', color='blue')
        plt.axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
        plt.title("Aggression (PFR)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 4. VPIP
    plt.subplot(3, 2, 4)
    if league_df is not None and not league_df.empty:
        plt.plot(league_df.index, league_df['VPIP'], label='VPIP %', color='purple')
        plt.axhspan(20, 30, color='green', alpha=0.2, label='Pro Zone')
        plt.title("Hand Selection (VPIP)")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 5. CPI / 3-Bet (Combined View)
    plt.subplot(3, 1, 3)
    if league_df is not None and not league_df.empty:
        # Plot 3-Bet on Left Axis
        ax1 = plt.gca()
        ax1.plot(league_df.index, league_df['ThreeBet_Pct'], color='orange', label='3-Bet %')
        ax1.set_ylabel('3-Bet %', color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')
        ax1.axhspan(6, 15, color='orange', alpha=0.1)
        
        # Plot CPI Score on Right Axis (if exists)
        if 'Score_Display' in league_df.columns:
            ax2 = ax1.twinx()
            ax2.plot(league_df.index, league_df['Score_Display'], color='black', linestyle='--', label='CPI Score')
            ax2.set_ylabel('CPI Score (0-1)', color='black')
            ax2.set_ylim(0, 1.0)
            
        plt.title("Advanced Metrics: 3-Bet & Performance Score")
        plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG)
    print(f"\n[SUCCESS] Enhanced charts saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    t_df = analyze_training()
    l_df = analyze_league()
    generate_charts(t_df, l_df)