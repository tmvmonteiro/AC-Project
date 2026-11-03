import pandas as pd

def clean_datasets(datasets):
    """
    Cleans all loaded datasets according to the defined exploratory cleaning rules.
    """
    # Remove unwanted datasets
    datasets_to_remove = ["TeamsSC", "GoaliesSC", "TeamsHalf", "CombinedShutouts", "abbrev"]
    for name in datasets_to_remove:
        datasets.pop(name, None)

    # AwardsCoaches - Remove 'note' column
    if 'AwardsCoaches' in datasets and 'note' in datasets['AwardsCoaches'].columns:
        datasets['AwardsCoaches'] = datasets['AwardsCoaches'].drop(columns=['note'])

    # AwardsMisc - Remove 'note' column; Remove rows without 'ID'
    if 'AwardsMisc' in datasets:
        df = datasets['AwardsMisc']
        if 'note' in df.columns:
            df = df.drop(columns=['note'])
        if 'ID' in df.columns:
            df = df.dropna(subset=['ID'])
        datasets['AwardsMisc'] = df

    # AwardsPlayers - Remove 'note' and 'pos' columns
    if 'AwardsPlayers' in datasets:
        df = datasets['AwardsPlayers']
        cols_to_drop = [col for col in ['note', 'pos'] if col in df.columns]
        datasets['AwardsPlayers'] = df.drop(columns=cols_to_drop)

    # Coaches - Remove 'note' column; delete empty rows or those missing 'g', 'w', 'l', 't'; add 0 to empty 'post' columns
    if 'Coaches' in datasets:
        df = datasets['Coaches']
        if 'note' in df.columns:
            df = df.drop(columns=['note'])
        essential_cols = [c for c in ['g', 'w', 'l', 't'] if c in df.columns]
        if essential_cols:
            df = df.dropna(subset=essential_cols)
        post_cols = [c for c in df.columns if c.startswith('post')]
        for col in post_cols:
            df[col] = df[col].fillna(0)
        datasets['Coaches'] = df

    # Goalies - Fill 0 for postseason stats; drop rows missing core identifiers (playerID or tmID)
    if 'Goalies' in datasets:
        df = datasets['Goalies']
        post_cols_goalies = ['PostGP', 'PostMin', 'PostW', 'PostL', 'PostT', 'PostENG', 'PostSHO', 'PostGA', 'PostSA']
        for col in post_cols_goalies:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        id_cols = [c for c in ['playerID', 'tmID'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        datasets['Goalies'] = df

    # GoaliesShootout - Fill 0 for missing shootout stats (W, L, SA, GA); drop rows missing playerID or tmID
    if 'GoaliesShootout' in datasets:
        df = datasets['GoaliesShootout']
        for col in ['W', 'L', 'SA', 'GA']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        id_cols = [c for c in ['playerID', 'tmID'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        datasets['GoaliesShootout'] = df

    # HOF - Remove rows missing essential identifiers (hofID or name); standardize category classifications
    if 'HOF' in datasets:
        df = datasets['HOF']
        id_cols = [c for c in ['hofID', 'name'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        if 'category' in df.columns:
            df['category'] = df['category'].str.strip().str.title()
        datasets['HOF'] = df

    # Master - Remove 'nameNote' column; drop rows missing primary player keys (playerID)
    if 'Master' in datasets:
        df = datasets['Master']
        if 'nameNote' in df.columns:
            df = df.drop(columns=['nameNote'])
        if 'playerID' in df.columns:
            df = df.dropna(subset=['playerID'])
        datasets['Master'] = df

    # Scoring - Fill 0 for specialty teams/postseason columns; drop rows missing playerID
    if 'Scoring' in datasets:
        df = datasets['Scoring']
        scoring_fill_cols = ['+/-', 'PPG', 'PPA', 'SHG', 'SHA', 'GWG', 'GTG', 'SOG', 
                             'PostGP', 'PostG', 'PostA', 'PostPts', 'PostPIM', 'Post+/-', 
                             'PostPPG', 'PostPPA', 'PostSHG', 'PostSHA', 'PostGWG', 'PostSOG']
        for col in scoring_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        if 'playerID' in df.columns:
            df = df.dropna(subset=['playerID'])
        datasets['Scoring'] = df

    # ScoringSC - Fill 0 for empty statistical columns (G, A, Pts, PIM) for Stanley Cup appearance players
    if 'ScoringSC' in datasets:
        df = datasets['ScoringSC']
        for col in ['G', 'A', 'Pts', 'PIM']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        datasets['ScoringSC'] = df

    # ScoringShootout - Fill 0 for empty shootout goals (G) and GDG; ensure valid counts for attempts (S)
    if 'ScoringShootout' in datasets:
        df = datasets['ScoringShootout']
        for col in ['G', 'GDG', 'S']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        datasets['ScoringShootout'] = df

    # ScoringSup - Fill 0 for missing supplemental power-play or short-handed assists (PPA, SHA)
    if 'ScoringSup' in datasets:
        df = datasets['ScoringSup']
        for col in ['PPA', 'SHA']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        datasets['ScoringSup'] = df

    # SeriesPost - Remove 'note' column; drop rows missing series round, winner ID, or loser ID
    if 'SeriesPost' in datasets:
        df = datasets['SeriesPost']
        if 'note' in df.columns:
            df = df.drop(columns=['note'])
        series_id_cols = [c for c in ['round', 'tmIDWinner', 'tmIDLoser'] if c in df.columns]
        if series_id_cols:
            df = df.dropna(subset=series_id_cols)
        datasets['SeriesPost'] = df

    # Teams - Fill 0 for pre-shootout era SoW/SoL; ensure correct integer types for rank and Pts
    if 'Teams' in datasets:
        df = datasets['Teams']
        for col in ['SoW', 'SoL']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        for col in ['rank', 'Pts']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        datasets['Teams'] = df

    # TeamSplits - Fill 0 for unplayed splits; drop rows missing core identifiers (tmID or year)
    if 'TeamSplits' in datasets:
        df = datasets['TeamSplits']
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        id_cols = [c for c in ['tmID', 'year'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        datasets['TeamSplits'] = df

    # TeamsPost - Fill 0 for postseason specialty stats; drop rows missing tmID or year
    if 'TeamsPost' in datasets:
        df = datasets['TeamsPost']
        post_team_cols = ['BenchMinor', 'PPG', 'PPC', 'SHA', 'PKG', 'PKC', 'SHF']
        for col in post_team_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        id_cols = [c for c in ['tmID', 'year'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        datasets['TeamsPost'] = df

    # TeamVsTeam - Fill 0 for missing head-to-head match stats (W, L, T, OTL); drop rows missing keys
    if 'TeamVsTeam' in datasets:
        df = datasets['TeamVsTeam']
        for col in ['W', 'L', 'T', 'OTL']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        id_cols = [c for c in ['tmID', 'oppID', 'year'] if c in df.columns]
        if id_cols:
            df = df.dropna(subset=id_cols)
        datasets['TeamVsTeam'] = df

    return datasets