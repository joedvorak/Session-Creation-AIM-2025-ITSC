import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Automated Session Creation Visualization Tool - AIM 2025 (ITSC Only)",
    page_icon=":material/category_search:",
    layout="wide",
)

def calculate_cluster_similarities(similarity_matrix, labels):
    """
    Calculate average similarity of each document with others in its cluster
    
    Returns:
    - document_similarities: Average similarity of each document with its cluster
    - cluster_avg_similarities: Average similarity for each cluster
    """
    n_samples = len(labels)
    document_similarities = np.zeros(n_samples)
    cluster_avg_similarities = {}
    
    for i in range(n_samples):
        # Get indices of other documents in the same cluster
        cluster_idx = np.where(labels == labels[i])[0]
        cluster_idx = cluster_idx[cluster_idx != i]  # Exclude self
        
        if len(cluster_idx) > 0:  # If there are other documents in the cluster
            # Calculate average similarity with other documents in cluster
            document_similarities[i] = np.mean(similarity_matrix[i, cluster_idx])
        
    # Calculate average similarity for each cluster
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        cluster_mask = labels == cluster
        cluster_docs = np.where(cluster_mask)[0]
        
        if len(cluster_docs) > 1:
            cluster_similarities = []
            for doc in cluster_docs:
                other_docs = cluster_docs[cluster_docs != doc]
                avg_sim = np.mean(similarity_matrix[doc, other_docs])
                cluster_similarities.append(avg_sim)
            cluster_avg_similarities[cluster] = np.mean(cluster_similarities)
        else:
            cluster_avg_similarities[cluster] = 0.0
            
    return document_similarities, cluster_avg_similarities

st.title("Automated Session Creation Visualization Tool - AIM 2025 (ITSC Only)")

def calculate_session_similarity(presentations, similarity_matrix, session_column='Original Session'):
    """
    Calculates the average similarity between presentations in different sessions.

    Args:
      presentations: A pandas DataFrame with presentation information.
      similarity_matrix: A NumPy array representing the pairwise similarity between presentations.
      session_column: The column name in the DataFrame that specifies the session.

    Returns:
      A tuple containing:
        - A NumPy array representing the session-level similarity matrix.
        - A NumPy array of unique session names in the order they appear in the similarity matrix.
    """
    # Create a new presentation dataframe that has constant indexing.
    presentations = presentations.reset_index()
    sessions = presentations[session_column].unique()
    num_sessions = len(sessions)
    session_similarity = np.zeros((num_sessions, num_sessions))

    for i, session1 in enumerate(sessions):
        for j, session2 in enumerate(sessions):
            pres_idx1 = presentations[presentations[session_column] == session1].index
            pres_idx2 = presentations[presentations[session_column] == session2].index
            session_similarity[i, j] = np.mean(similarity_matrix[np.ix_(pres_idx1, pres_idx2)])

    return session_similarity, sessions

def analyze_sessions(sessions_dict, similarities):
    """
    Calculates statistics for sessions using a dictionary and a NumPy similarity matrix.

    Args:
        sessions_dict (dict): A dictionary where keys are session names 
                              and values are lists of indices of items in that session.
        similarities (np.ndarray): The cosine similarity matrix as a NumPy array.

    Returns:
        dict: A dictionary containing the calculated statistics:
              - 'avg_similarity': Overall average similarity across sessions with > 1 item.
              - 'min_session_similarity': Minimum average similarity found in any session with > 1 item.
              - 'num_sessions': Total number of sessions.
              - 'clusters_gt1': Number of sessions with more than one item.
              - 'session_sizes': A dictionary mapping session names to their sizes.
              - 'similarity_values': A dictionary mapping session names to their average internal similarity.
    """
    # Initialize dictionaries to store results for each session
    session_similarities = {}
    session_sizes = {}
    clusters_gt1 = 0 # Counter for sessions with more than one item

    # Iterate through each session in the input dictionary
    for session_name, session_indices in sessions_dict.items():
        # Store the size (number of items) of the current session
        session_sizes[session_name] = len(session_indices)
        
        # Calculate similarity only for sessions with more than one item
        if len(session_indices) > 1:
            clusters_gt1 += 1 # Increment the counter for sessions > size 1
            
            # Use numpy advanced indexing (np.ix_) to extract the submatrix 
            # corresponding to the current session's indices
            session_similarity_matrix = similarities[np.ix_(session_indices, session_indices)]
            
            # Calculate the sum of the upper triangle of the submatrix (excluding the diagonal)
            # This sums the similarities between unique pairs within the session
            similarity_sum = np.sum(np.triu(session_similarity_matrix, k=1))
            
            # Calculate the number of unique pairs in the session
            num_pairs = len(session_indices) * (len(session_indices) - 1) / 2
            
            # Calculate the average similarity for the session
            # Handle the case where num_pairs might be zero (although guarded by len > 1 check)
            avg_session_similarity = similarity_sum / num_pairs if num_pairs > 0 else 0.0
            
            # Store the calculated average similarity for the session
            session_similarities[session_name] = avg_session_similarity
            
        # For sessions with only one item, similarity is not meaningful (set to 0)
        elif len(session_indices) == 1:
            session_similarities[session_name] = 0.0 
        # For empty sessions, similarity is 0
        else:
            session_similarities[session_name] = 0.0 

    # Calculate the overall average similarity and minimum similarity across sessions with > 1 item.
    # Iterate through the calculated similarities, checking the original session size.
    valid_similarities = [
        sim for session_name, sim in session_similarities.items() 
        if len(sessions_dict.get(session_name, [])) > 1 
    ] # Filter to include only similarities from sessions with more than one item.
    
    overall_avg_similarity = np.mean(valid_similarities) if valid_similarities else 0.0
    min_session_similarity = min(valid_similarities) if valid_similarities else 0.0

    # Compile the results into a dictionary
    results = {
        'avg_similarity': overall_avg_similarity,
        'min_session_similarity': min_session_similarity,
        'num_sessions': len(sessions_dict),
        'clusters_gt1': clusters_gt1,
        'session_sizes': session_sizes, # Dictionary of session_name: size
        'similarity_values': session_similarities, # Dictionary of session_name: avg_similarity
    }

    return results

def create_session_index_lists(oral_df, session_column='Original Session'):
    """
    Creates a list of lists, where each inner list contains the indices of presentations
    assigned to a particular session.

    Args:
        oral_df (pd.DataFrame): The DataFrame containing presentation data.
        session_column (str): The name of the column containing session names.

    Returns:
        list: A list of lists, where each inner list contains indices for a session.
        dict: A dictionary mapping session names to their corresponding indices.
    """
    # Reset index to ensure consistent indexing (similarity tensor is re-indexed)
    oral_df.reset_index(drop=True, inplace=True)  
    session_indices = {}
    for index, session in oral_df[session_column].items():
        if session not in session_indices:
            session_indices[session] = []
        session_indices[session].append(index)

    return list(session_indices.values()), session_indices

st.subheader("Data Retrieval Date: January 29, 2025")
st.write("**NOTE:** ***This sort is based on the initial submission of titles, abstracts and session placement before session organization process.***")
st.write("This only includes presentations submitted to the ITSC technical community for AIM 2025. This is a companion tool to a research manusciprt that is currently under review.")

# Load DataFrames
# Load presentations first.
df_presentations = pd.read_pickle("itsc_clustered_presentations_no_abstracts.pkl")
# Drop the unnecessary "index" column if it exists
if 'index' in df_presentations.columns:
    df_presentations = df_presentations.drop(columns=['index'])
# Load sessions next
df_AI_sessions = pd.read_pickle("AI_sessions.pkl")
# Load the similarity matrix
df_similarity = pd.read_pickle("itsc_oral_app_similarities.pkl")

# Calculate statistics for the original session placements
# Create a session similarity matrix that contains the similarities from the original sessions.
orig_session_similarity_matrix, sessions = calculate_session_similarity(df_presentations, df_similarity.to_numpy(), session_column='Original Session')
df_orig_session_similarity = pd.DataFrame(orig_session_similarity_matrix, index=sessions, columns=sessions)

pres_orig_session_similarity, orig_session_similarity = calculate_cluster_similarities(df_similarity.to_numpy(), np.array(df_presentations['Original Session']))
original_session_list, original_session_dict = create_session_index_lists(df_presentations, session_column='Original Session')
original_analysis_results = analyze_sessions(original_session_dict, df_similarity.to_numpy())
df_presentations['Original Presentation-Session Similarity'] = pres_orig_session_similarity
df_presentations['Original Session Similarity'] = df_presentations['Original Session'].map(orig_session_similarity)

# Calculate standard deviation for each 'Original Session'
df_presentations['Original Session Std Dev'] = df_presentations.groupby('Original Session')['Original Presentation-Session Similarity'].transform('std')
# Calculate deviation from the mean
df_presentations['Original Raw Deviation'] = df_presentations['Original Presentation-Session Similarity'] - df_presentations['Original Session Similarity']
# Calculate standardized deviation
df_presentations['Original Standardized Deviation'] = df_presentations['Original Raw Deviation'] / df_presentations['Original Session Std Dev']

# Create the Original Sessions DataFrame
df_sessions = df_presentations[['Original Session', 'Original Session Similarity', 'Original Session Std Dev']].copy()
# Drop duplicate rows, keeping only the first instance, then reindex.
df_sessions.drop_duplicates(subset=['Original Session'], keep='first', inplace=True)
df_sessions = df_sessions.reset_index(drop=True)

# Calculate statistics for the clustering session placements
# Create a session similarity matrix that contains the similarities from the clustering sessions.
clustering_session_similarity_matrix, clustering_sessions = calculate_session_similarity(df_presentations, df_similarity.to_numpy(), session_column='Clustering Session')
df_clustering_session_similarity = pd.DataFrame(clustering_session_similarity_matrix, index=clustering_sessions, columns=clustering_sessions)

pres_clustering_session_similarity, clustering_session_similarity = calculate_cluster_similarities(df_similarity.to_numpy(), np.array(df_presentations['Clustering Session']))
cluster_session_list, cluster_session_dict = create_session_index_lists(df_presentations, session_column='Clustering Session')
cluster_analysis_results = analyze_sessions(cluster_session_dict, df_similarity.to_numpy())
df_presentations['Presentation-Session Similarity - Clustering'] = pres_clustering_session_similarity
df_presentations['Session Similarity - Clustering'] = df_presentations['Clustering Session'].map(clustering_session_similarity)

# Calculate standard deviation for each 'Clustering Session'
df_presentations['Session Std Dev - Clustering'] = df_presentations.groupby('Clustering Session')['Presentation-Session Similarity - Clustering'].transform('std')
# Calculate deviation from the mean
df_presentations['Raw Deviation - Clustering'] = df_presentations['Presentation-Session Similarity - Clustering'] - df_presentations['Session Similarity - Clustering']
# Calculate standardized deviation
df_presentations['Standardized Deviation - Clustering'] = df_presentations['Raw Deviation - Clustering'] / df_presentations['Session Std Dev - Clustering']

# Create the Clustering Sessions DataFrame
df_clustering_sessions = df_presentations[['Clustering Session', 'Session Similarity - Clustering', 'Session Std Dev - Clustering']].copy()
# Drop duplicate rows, keeping only the first instance, then reindex.
df_clustering_sessions.drop_duplicates(subset=['Clustering Session'], keep='first', inplace=True)
df_clustering_sessions = df_clustering_sessions.reset_index(drop=True)
# Exclude 'Indices' and 'Presentations' columns from df_AI_sessions
columns_to_merge = [col for col in df_AI_sessions.columns if col not in ['Indices', 'Presentations']]

# Merge the data from df_AI_sessions into df_clustering_sessions
df_clustering_sessions = df_clustering_sessions.merge(
    df_AI_sessions[columns_to_merge], 
    left_on='Clustering Session', 
    right_index=True, 
    how='left'
)

# Update column names for display
df_clustering_sessions = df_clustering_sessions.rename(
    columns=lambda col: col.replace(" MinEx ", " Minimal Example ")
                           .replace(" NoEx ", " No Example ")
                           .replace(" Ex ", " Complete Example ")  # Add a space before and after "Ex" to avoid partial matches
)


tab_clustered_session, tab_orig_session, tab_pres =  st.tabs(['View Clustered Sessions', 'View Original Sessions', 'View Presentations'])

with tab_clustered_session:
    st.header("Clustered Sessions")
    st.write(f"Average Session Similarity*: {cluster_analysis_results['avg_similarity']:.3f}")
    st.write(f"Minimum Session Similarity*: {cluster_analysis_results['min_session_similarity']:.3f}")
    st.write(f"Number of Sessions: {cluster_analysis_results['num_sessions']}")
    st.write(f"Number of sessions with more than 1 item: {cluster_analysis_results['clusters_gt1']}")
    st.write(f"Presentations placed in sessions: {sum([len(s) for s in cluster_session_list])}")
    st.write(f"*Values do not include sessions with only one item as they cannot have a meaningful similarity score.")
    
    # Plot session size 
    # Get the session sizes dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the sizes.
    cluster_session_sizes_dict = cluster_analysis_results['session_sizes']
    cluster_session_sizes_df = pd.DataFrame.from_dict(
        cluster_session_sizes_dict, 
        orient='index', 
        columns=['Presentations Submitted'] # This sets the column name for the bar heights
    )
    cluster_session_sizes_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sizes_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Size Distribution - Clustered Sessions") 
    st.bar_chart(cluster_session_sizes_df, x_label="Session Name", y_label="Presentations Count") 

    # Plot session similarity distribution
    # Get the session similarity dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the similarities.
    cluster_session_sim_dict = cluster_analysis_results['similarity_values']
    cluster_session_sim_df = pd.DataFrame.from_dict(
        cluster_session_sim_dict, 
        orient='index', 
        columns=['Session Similiarity'] # This sets the column name for the bar heights
    )
    cluster_session_sim_df.index.name = 'Session Name'  # Set the index name for better labeling
    cluster_session_sim_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Similarity Distribution - Clustered Sessions") 
    st.bar_chart(cluster_session_sim_df, x_label="Session Name", y_label="Session Similiarity")

    with st.expander('**Instructions** Click to expand'):
        st.write("Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table.")
    df_clustering_sessions.set_index('Clustering Session', inplace=True)
    event_clustered_session = st.dataframe(
            df_clustering_sessions,
            use_container_width=True,
            column_order=["Clustering Session", 'Session Similarity - Clustering', 'Session Std Dev - Clustering'],
            column_config={
                "Session Similarity - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                "Session Std Dev - Clustering" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    if event_clustered_session.selection.rows: # Check if a session has been selected.
        selected_clustered_session_df = df_clustering_sessions.iloc[event_clustered_session.selection.rows]  # Create a dataframe from the selected session row.
        selected_clustered_session = selected_clustered_session_df.index[0]
        st.header(f"Session {int(selected_clustered_session):d} Session Details")
        selected_clustered_session_df_transposed = selected_clustered_session_df.T
        selected_clustered_session_df_transposed.drop(index=['Session Similarity - Clustering', 'Session Std Dev - Clustering'], inplace=True)

        st.write(f"**Session Similarity:** {selected_clustered_session_df.iloc[0]['Session Similarity - Clustering']:.3f}")
        st.write(f"**Session Similarity Std Dev:** {selected_clustered_session_df.iloc[0]['Session Std Dev - Clustering']:.3f}")
        with st.expander("**Table Details** Click to expand"):
            st.write("This table shows the LLM output for the session. They are named as [LLM][Prompt][Variable].")
            st.write("For example: *Gemini Complete Example Title 1* is the 1st title option provided by the Gemini LLM for the Complete Example prompt.")
        st.dataframe(
            selected_clustered_session_df_transposed,
            use_container_width=True,
        )
        st.write(f"**Presentations in this session**")
        df_selected_clustered_session = df_presentations[df_presentations['Clustering Session'] == selected_clustered_session]
        st.dataframe(
            df_selected_clustered_session,
            use_container_width=True,
            hide_index=True,
            column_order=["Presentation-Session Similarity - Clustering", 'Standardized Deviation - Clustering', 'Abstract ID', 'Title', 'Original Session'],
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                "Session Similarity - Clustering" : None,
                'Session Std Dev - Clustering': None,
                "Raw Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
                "Standardized Deviation - Clustering" : st.column_config.NumberColumn(format='%.3f'),
            },
        )
        st.header("Most Similar Sessions")
        # Create a Series with the most similar sessions
        similar_clustered_sessions = df_clustering_session_similarity[selected_clustered_session].sort_values(ascending=False) 
        # Remove the selected session itself from the similar sessions
        similar_clustered_sessions = similar_clustered_sessions.drop(selected_clustered_session)
        similar_clustered_sessions_df = pd.DataFrame(similar_clustered_sessions)
        st.write("Other sessions that are most similar to:")
        st.subheader(f"Session {int(similar_clustered_sessions_df.columns[0]):d}")
        st.write("This list is initially sorted by similarity to the selected session.")
        similar_clustered_sessions_df = similar_clustered_sessions_df.rename(columns={
            similar_clustered_sessions_df.columns[0]:'Session-Session Similarity Score',
            })
        similar_clustered_sessions_df.insert(0, "Session Similarity Rank", np.arange(1,similar_clustered_sessions_df.shape[0]+1))
        st.dataframe(
            similar_clustered_sessions_df,
            use_container_width=True,
            hide_index=False,
            )

with tab_orig_session:
    st.header("Original Sessions")
    st.write(f"Average Session Similarity*: {original_analysis_results['avg_similarity']:.3f}")
    st.write(f"Minimum Session Similarity*: {original_analysis_results['min_session_similarity']:.3f}")
    st.write(f"Number of Sessions: {original_analysis_results['num_sessions']}")
    st.write(f"Number of sessions with more than 1 item: {original_analysis_results['clusters_gt1']}")
    st.write(f"Presentations placed in sessions: {sum([len(s) for s in original_session_list])}")
    st.write(f"*Values do not include sessions with only one item as they cannot have a meaningful similarity score.")
    
    # Plot session size 
    # Get the session sizes dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the sizes.
    orig_session_sizes_dict = original_analysis_results['session_sizes']
    orig_session_sizes_df = pd.DataFrame.from_dict(
        orig_session_sizes_dict, 
        orient='index', 
        columns=['Presentations Submitted'] # This sets the column name for the bar heights
    )
    orig_session_sizes_df.index.name = 'Session Name'  # Set the index name for better labeling
    orig_session_sizes_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Size Distribution - Original Sessions") 
    st.bar_chart(orig_session_sizes_df, x_label="Session Name", y_label="Presentations Count") 

    # Plot session similarity distribution
    # Get the session similarity dictionary, convert to a DataFrame for better labeling in st.bar_chart
    # The index will be the session names, and the values will be the similarities.
    orig_session_sim_dict = original_analysis_results['similarity_values']
    orig_session_sim_df = pd.DataFrame.from_dict(
        orig_session_sim_dict, 
        orient='index', 
        columns=['Session Similiarity'] # This sets the column name for the bar heights
    )
    orig_session_sim_df.index.name = 'Session Name'  # Set the index name for better labeling
    orig_session_sim_df.sort_index(inplace=True)  # Sort the index for better visualization
    st.subheader("Session Similarity Distribution - Original Sessions") 
    st.bar_chart(orig_session_sim_df, x_label="Session Name", y_label="Session Similiarity")

    with st.expander('**Instructions** Click to expand'):
        st.write("Select a session by clicking on the checkbox in the leftmost column. Its details and assigned presentations will appear below. You can sort the session list by any column or search for a session name. Just click on the column or mouse over the table.")
    event_session = st.dataframe(
            df_sessions,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Original Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Original Session Std Dev" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )

    if event_session.selection.rows: # Check if a session has been selected.
        st.header('Session Details')
        selected_session_df = df_sessions.iloc[event_session.selection.rows]  # Create a dataframe from the selected session row.
        selected_session =selected_session_df.iloc[0]['Original Session']
        st.subheader(selected_session)
        st.write(f"**Session Similarity:** {selected_session_df.iloc[0]['Original Session Similarity']:.3f}")
        df_selected_session = df_presentations[df_presentations['Original Session'] == selected_session]
        st.dataframe(
            df_selected_session,
            use_container_width=True,
            hide_index=True,
            column_order=["Original Presentation-Session Similarity", 'Original Standardized Deviation', 'Abstract ID', 'Title', ],
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Original Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Original Session Similarity" : None,
                'Original Session Std Dev': None,
                "Original Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                "Original Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
            },
        )
        st.header("Most Similar Sessions")
        # Create a Series with the  most similar sessions
        similar_sessions = df_orig_session_similarity[selected_session].sort_values(ascending=False) 
        # Remove the selected presentation itself from the similar presentations
        similar_sessions = similar_sessions.drop(selected_session)
        similar_sessions_df = pd.DataFrame(similar_sessions)
        st.write("Other sessions that are most similar to:")
        st.subheader(similar_sessions_df.columns[0])
        st.write("This list is initially sorted by similarity to the selected session.")
        similar_sessions_df = similar_sessions_df.rename(columns={
            similar_sessions_df.columns[0]:'Session-Session Similarity Score',
            })
        similar_sessions_df.insert(0, "Session Similarity Rank", np.arange(1,similar_sessions_df.shape[0]+1))
        st.dataframe(
            similar_sessions_df,
            use_container_width=True,
            hide_index=False,
            )

with tab_pres:
    st.header("Presentations") 
    with st.expander('**Instructions** Click to expand'):
        st.write("Select a presentation by clicking on the checkbox. You can sort the presentation list or search as well.")
        st.write("Once a presentation is selected, its abstract and the ten most similar presentations will appear in a list below.")
        st.write("If you move your mouse over the table, a menu will appear in the top left corner that lets you search within the table or download. Clicking on columns will let you sort by the column too.")
        st.write("If text is cut off, click twice on an cell to see the full text. You can scroll left-right and up-down in the table.")
        st.write("Similarity scores range from 0.0 (not similar) to 1.0 (identical).")
        st.write("The leftmost column is a checkbox column. Click to select a presentation. This may blend with the background on dark themes.")

    event = st.dataframe(
            df_presentations,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Original Presentation-Session Similarity" : st.column_config.NumberColumn(format='%.3f'),
                "Original Session Similarity" : None,
                'Original Session Std Dev': None,
                "Original Raw Deviation" : st.column_config.NumberColumn(format='%.3f'),
                "Original Standardized Deviation" : st.column_config.NumberColumn(format='%.3f'),
            },
            on_select="rerun",
            selection_mode="single-row",
        )


    if event.selection.rows: # Check if a presentation has been selected.
        st.header("Selected Presentation:")
        selected_pres = df_presentations.iloc[event.selection.rows]  # Create a dataframe from the selected presentation row.
        st.write(selected_pres.iloc[0]['Title'])  # It is necessary to request the first row, [0], since it is a dataframe and not just one entry.
        st.header("Most Similar Presentations")
        similar_presentations = df_similarity.loc[selected_pres.iloc[0].name].sort_values(ascending=False) # Create a Series with the  most similar presentations
        # Remove the selected presentation itself from the similar presentations
        similar_presentations = similar_presentations.drop(selected_pres.iloc[0].name)
        # Build the similarity dataframe. Add the similarity score and similarity rank to the dataframe and show it.
        similar_df = df_presentations.loc[similar_presentations.index]
        similar_df.insert(0, "Similarity Score", similar_presentations)
        similar_df.insert(0, "Similarity Rank", np.arange(1,similar_df.shape[0]+1))
        st.dataframe(
            similar_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Abstract ID' : st.column_config.NumberColumn(format='%i'),
                "Presentation-Session Similarity" : None,
                "Session Similarity" : None,
                'Session Std Dev': None,
                "Raw Deviation" : None,
                "Standardized Deviation" : None,
            },
            )