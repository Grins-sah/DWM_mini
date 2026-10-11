import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, ArrowDownRight, ArrowRight, Bell, Check, ChevronDown, Clapperboard,
  Compass, Film, Flame, Heart, Layers3, LoaderCircle, LogIn, Menu, Plus,
  RefreshCw, Search, Send, Settings2, Sparkles, Star, ThumbsDown, ThumbsUp,
  UserRound, Users, WandSparkles, X
} from "lucide-react";
import "./styles.css";

const API_BASE_DEFAULT = "http://127.0.0.1:8000";
const FALLBACK_GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"];
const INITIAL_PREFS = { Action: 4, Adventure: 3, Comedy: 3, Drama: 3, Romance: 2, "Sci-Fi": 5, Thriller: 4 };

function App() {
  const [apiBase, setApiBase] = useState(localStorage.getItem("reelcluster_api") || API_BASE_DEFAULT);
  const [userId, setUserId] = useState(localStorage.getItem("reelcluster_user") || "");
  const [user, setUser] = useState(null);
  const [genres, setGenres] = useState(FALLBACK_GENRES);
  const [preferences, setPreferences] = useState(INITIAL_PREFS);
  const [recommendations, setRecommendations] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [movies, setMovies] = useState([]);
  const [activeTab, setActiveTab] = useState("discover");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [busyAction, setBusyAction] = useState("");
  const [toast, setToast] = useState("");
  const [apiError, setApiError] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(!localStorage.getItem("reelcluster_user"));
  const [showNewMovie, setShowNewMovie] = useState(false);
  const [movieForm, setMovieForm] = useState({ movie_id: "", title: "", genres: ["Action", "Sci-Fi"], description: "", min_cluster_score: 3.5, max_clusters: 3 });
  const [mobileMenu, setMobileMenu] = useState(false);

  const api = async (path, options = {}) => {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    const raw = await response.text();
    let data = {};
    try { data = raw ? JSON.parse(raw) : {}; } catch { data = { detail: raw }; }
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
      throw new Error(message || `Request failed (${response.status})`);
    }
    return data;
  };

  const notify = (message) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 3200);
  };

  const loadGenres = async () => {
    try {
      const data = await api("/genres");
      if (data.genres?.length) setGenres(data.genres);
    } catch (error) {
      setApiError(error.message);
    }
  };

  const loadUserData = async (id = userId) => {
    if (!id.trim()) return;
    setLoading(true);
    setApiError("");
    try {
      const [profile, recs, notes] = await Promise.all([
        api(`/users/${encodeURIComponent(id)}`),
        api(`/users/${encodeURIComponent(id)}/recommendations?limit=30`),
        api(`/users/${encodeURIComponent(id)}/notifications`),
      ]);
      setUser(profile);
      setPreferences(profile.preferences || {});
      setRecommendations(recs.recommendations || []);
      setNotifications(notes.notifications || []);
      localStorage.setItem("reelcluster_user", id);
      setUserId(id);
      setShowOnboarding(false);
    } catch (error) {
      setApiError(error.message);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const loadMovies = async (q = "") => {
    try {
      const data = await api(`/movies?limit=100${q ? `&q=${encodeURIComponent(q)}` : ""}`);
      setMovies(data || []);
    } catch (error) {
      // The catalogue endpoint lists movies added to the app DB; recommendations use the model catalogue.
      setMovies([]);
    }
  };

  useEffect(() => {
    loadGenres();
    if (userId) loadUserData(userId);
    // API connection is intentionally initialized once on page load.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (activeTab === "catalogue") loadMovies(search);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  const connect = async () => {
    setLoading(true);
    setApiError("");
    try {
      const health = await api("/health");
      if (!health.model_loaded) throw new Error("API is reachable, but the K-Means model is not loaded. Train the model and check artifacts/movie_model.joblib.");
      await loadGenres();
      if (userId) await loadUserData(userId);
      notify("Connected to recommendation API");
    } catch (error) {
      setApiError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const createUser = async () => {
    const id = userId.trim();
    if (!id) { setApiError("Enter a user ID to continue."); return; }
    setLoading(true);
    setApiError("");
    try {
      const result = await api("/users", {
        method: "POST",
        body: JSON.stringify({ user_id: id, preferences }),
      });
      localStorage.setItem("reelcluster_user", id);
      setUserId(id);
      setUser({ user_id: id, cluster_id: result.cluster_id, preferences });
      setRecommendations(result.recommendations || []);
      setShowOnboarding(false);
      notify(`Welcome in — assigned to cluster ${result.cluster_id}`);
      await loadUserData(id);
    } catch (error) {
      if (error.message.toLowerCase().includes("already exists")) {
        await loadUserData(id);
        notify("Welcome back");
      } else setApiError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const savePreferences = async () => {
    if (!userId) return;
    setBusyAction("preferences");
    try {
      const result = await api(`/users/${encodeURIComponent(userId)}`, {
        method: "PUT",
        body: JSON.stringify({ preferences }),
      });
      setUser((old) => ({ ...old, cluster_id: result.cluster_id, preferences }));
      await loadUserData(userId);
      notify("Taste profile updated");
    } catch (error) { setApiError(error.message); }
    finally { setBusyAction(""); }
  };

  const recordInteraction = async (movie, event_type, rating = null) => {
    if (!userId) { setShowOnboarding(true); return; }
    setBusyAction(`${movie.movie_id}-${event_type}`);
    try {
      await api(`/users/${encodeURIComponent(userId)}/interactions`, {
        method: "POST",
        body: JSON.stringify({ movie_id: String(movie.movie_id), event_type, rating }),
      });
      await loadUserData(userId);
      notify(event_type === "like" ? "Added to your taste signals" : event_type === "watchlist" ? "Saved to your watchlist" : "Interaction recorded");
    } catch (error) { setApiError(error.message); }
    finally { setBusyAction(""); }
  };

  const markRead = async (note) => {
    try {
      await api(`/notifications/${note.notification_id}`, {
        method: "PATCH",
        body: JSON.stringify({ read: true }),
      });
      setNotifications((old) => old.map((item) => item.notification_id === note.notification_id ? { ...item, read: 1 } : item));
    } catch (error) { setApiError(error.message); }
  };

  const submitMovie = async () => {
    setBusyAction("new-movie");
    setApiError("");
    try {
      const result = await api("/movies", {
        method: "POST",
        body: JSON.stringify(movieForm),
      });
      notify(`Movie added · ${result.notifications_created} notifications created`);
      setShowNewMovie(false);
      setMovieForm({ movie_id: "", title: "", genres: ["Action", "Sci-Fi"], description: "", min_cluster_score: 3.5, max_clusters: 3 });
      await loadUserData(userId);
    } catch (error) { setApiError(error.message); }
    finally { setBusyAction(""); }
  };

  const visibleRecommendations = useMemo(() => {
    const q = search.toLowerCase().trim();
    return recommendations.filter((movie) =>
      !q || movie.title?.toLowerCase().includes(q) || movie.genres?.some((genre) => genre.toLowerCase().includes(q))
    );
  }, [recommendations, search]);

  const unreadCount = notifications.filter((note) => !note.read).length;
  const genreEmoji = (movie) => {
    const list = movie.genres || [];
    if (list.includes("Sci-Fi")) return "✦";
    if (list.includes("Horror")) return "☾";
    if (list.includes("Romance")) return "♡";
    if (list.includes("Comedy")) return "☻";
    if (list.includes("Action")) return "⚡";
    return "✧";
  };
  const palette = ["poster-violet", "poster-blue", "poster-orange", "poster-green", "poster-pink", "poster-gold"];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Clapperboard size={21} strokeWidth={2.2} /></div>
          <span>reel<span className="brand-accent">cluster</span></span>
          <button className="icon-button sidebar-close" onClick={() => setMobileMenu(false)} aria-label="Close menu"><X size={18}/></button>
        </div>
        <div className="workspace-label">YOUR SPACE</div>
        <nav className="nav-list">
          <button className={`nav-item ${activeTab === "discover" ? "active" : ""}`} onClick={() => {setActiveTab("discover"); setMobileMenu(false);}}><Compass size={18}/> Discover <span className="nav-dot"/></button>
          <button className={`nav-item ${activeTab === "notifications" ? "active" : ""}`} onClick={() => {setActiveTab("notifications"); setMobileMenu(false);}}><Bell size={18}/> Notifications {unreadCount > 0 && <span className="nav-count">{unreadCount}</span>}</button>
          <button className={`nav-item ${activeTab === "catalogue" ? "active" : ""}`} onClick={() => {setActiveTab("catalogue"); setMobileMenu(false);}}><Film size={18}/> My catalogue</button>
          <button className={`nav-item ${activeTab === "taste" ? "active" : ""}`} onClick={() => {setActiveTab("taste"); setMobileMenu(false);}}><SlidersIcon/> Taste profile</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="cluster-card">
            <div className="cluster-card-top"><span className="pulse"/><span>YOUR CLUSTER</span><Layers3 size={15}/></div>
            <div className="cluster-number">{user ? String(user.cluster_id).padStart(2, "0") : "—"}</div>
            <div className="cluster-caption">{user ? "Personalized from your taste" : "Connect to discover yours"}</div>
            <div className="cluster-bars"><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/><i/></div>
          </div>
          <button className="nav-item settings-link" onClick={() => setShowSettings(true)}><Settings2 size={18}/> API settings</button>
          <div className="profile-mini">
            <div className="avatar"><UserRound size={18}/></div>
            <div className="profile-mini-text"><strong>{user?.user_id || "Guest explorer"}</strong><span>{user ? `Cluster ${user.cluster_id} member` : "Not signed in"}</span></div>
            <button className="icon-button" onClick={() => setShowSettings(true)} aria-label="Settings"><ChevronDown size={16}/></button>
          </div>
        </div>
      </aside>

      {mobileMenu && <button className="mobile-scrim" onClick={() => setMobileMenu(false)} aria-label="Close menu"/>}
      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button mobile-menu-button" onClick={() => setMobileMenu(true)} aria-label="Open menu"><Menu size={20}/></button>
          <div className="breadcrumb"><span>Workspace</span><span className="crumb-slash">/</span><strong>{activeTab === "discover" ? "Discover" : activeTab === "notifications" ? "Notifications" : activeTab === "catalogue" ? "My catalogue" : "Taste profile"}</strong></div>
          <div className="topbar-actions">
            <button className="icon-button notification-button" onClick={() => setActiveTab("notifications")} aria-label="Notifications"><Bell size={18}/>{unreadCount > 0 && <i/>}</button>
            <div className="topbar-divider"/>
            <button className="user-chip" onClick={() => setShowSettings(true)}><span className="avatar avatar-small"><UserRound size={15}/></span><span>{user?.user_id || "Guest"}</span><ChevronDown size={14}/></button>
          </div>
        </header>

        <div className="page-content">
          {apiError && <div className="alert"><Activity size={17}/><div><strong>API notice</strong><span>{apiError}</span></div><button onClick={() => setApiError("")} aria-label="Dismiss"><X size={16}/></button></div>}

          {activeTab === "discover" && <>
            <section className="hero">
              <div className="hero-copy">
                <div className="eyebrow"><Sparkles size={14}/> YOUR NEXT OBSESSION AWAITS</div>
                <h1>Less scrolling.<br/><span>More feeling.</span></h1>
                <p>A little machine learning, a lot of movie magic. Your taste is unique — your recommendations should be too.</p>
                <div className="hero-actions">
                  <button className="button button-primary" onClick={() => document.getElementById("recommendations")?.scrollIntoView({ behavior: "smooth" })}>Explore your picks <ArrowRight size={16}/></button>
                  <button className="button button-quiet" onClick={() => setShowOnboarding(true)}><WandSparkles size={16}/> Tune my taste</button>
                </div>
                <div className="hero-proof"><div className="proof-avatars"><span>G</span><span>M</span><span>✦</span></div><span><strong>K-Means powered</strong> · grouped by what you love</span></div>
              </div>
              <div className="hero-art">
                <div className="orbit orbit-one"/><div className="orbit orbit-two"/>
                <div className="hero-poster poster-main"><div className="poster-glow"/><div className="poster-sun"/><div className="poster-mountains"/><span className="poster-kicker">A REELCLUSTER ORIGINAL</span><div className="poster-title">BEYOND<br/>THE <em>STARS</em></div><span className="poster-bottom">SOMEWHERE, A STORY FINDS YOU</span></div>
                <div className="floating-card floating-match"><div className="match-icon"><Sparkles size={16}/></div><div><strong>94% taste match</strong><span>Based on your cluster</span></div></div>
                <div className="floating-card floating-new"><span className="new-dot"/> New for your cluster <ArrowDownRight size={15}/></div>
                <div className="hero-star star-one">✦</div><div className="hero-star star-two">✧</div><div className="hero-star star-three">✦</div>
              </div>
            </section>

            <section className="stats-row">
              <div className="stat-card"><div className="stat-icon stat-purple"><Layers3 size={17}/></div><div><span className="stat-label">YOUR CLUSTER</span><strong>{user ? `#${user.cluster_id}` : "Not assigned"}</strong><small>{user ? "Taste group identified" : "Complete onboarding"}</small></div></div>
              <div className="stat-card"><div className="stat-icon stat-orange"><Film size={17}/></div><div><span className="stat-label">PICKS FOR YOU</span><strong>{recommendations.length || "—"}</strong><small>From the movie catalogue</small></div></div>
              <div className="stat-card"><div className="stat-icon stat-green"><Bell size={17}/></div><div><span className="stat-label">NEW DISCOVERIES</span><strong>{unreadCount}</strong><small>Unread notifications</small></div></div>
              <div className="stat-card"><div className="stat-icon stat-blue"><Activity size={17}/></div><div><span className="stat-label">TASTE SIGNALS</span><strong>{Object.keys(user?.preferences || {}).length || "—"}</strong><small>Genres in your profile</small></div></div>
            </section>

            <section className="section-block" id="recommendations">
              <div className="section-heading"><div><div className="eyebrow eyebrow-muted"><Flame size={14}/> CURATED FOR YOUR CLUSTER</div><h2>Your next watch</h2><p>Picked from movies rated highly by people with similar taste.</p></div><div className="section-controls"><div className="search-box"><Search size={16}/><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search movies or genres"/></div><button className="button button-outline" onClick={() => loadUserData()} disabled={loading}><RefreshCw size={15}/> Refresh</button></div></div>
              {!user && <div className="empty-state"><div className="empty-icon"><WandSparkles size={24}/></div><h3>Your personal cinema starts here</h3><p>Set your genre preferences to get assigned to a K-Means cluster and unlock recommendations.</p><button className="button button-primary" onClick={() => setShowOnboarding(true)}>Build my taste profile <ArrowRight size={16}/></button></div>}
              {user && visibleRecommendations.length === 0 && <div className="empty-state"><div className="empty-icon"><Film size={24}/></div><h3>No picks found just yet</h3><p>Try a different search or refresh your recommendations.</p><button className="button button-outline" onClick={() => {setSearch(""); loadUserData();}}>Reset and refresh</button></div>}
              <div className="movie-grid">
                {visibleRecommendations.map((movie, index) => <MovieCard key={movie.movie_id} movie={movie} index={index} palette={palette} genreEmoji={genreEmoji} busyAction={busyAction} onAction={recordInteraction}/>)}
              </div>
              {user && <div className="section-footnote"><span className="pulse"/><span>Recommendations are ranked from ratings within cluster <strong>{user.cluster_id}</strong>. Interactions help refine your taste profile.</span></div>}
            </section>

            <section className="bottom-banner"><div className="banner-symbol"><Sparkles size={22}/></div><div><span className="eyebrow">GOT A NEW RELEASE?</span><h3>Find the audience that will love it.</h3><p>Add a movie and ReelCluster will identify matching preference groups.</p></div><button className="button button-primary" onClick={() => setShowNewMovie(true)}><Plus size={16}/> Add a movie <ArrowRight size={15}/></button></section>
          </>}

          {activeTab === "notifications" && <section className="subpage"><div className="subpage-heading"><div className="eyebrow eyebrow-muted"><Bell size={14}/> YOUR PERSONAL FEED</div><h1>Notifications</h1><p>New movies selected for your taste cluster.</p></div><div className="notification-list">
            {!notifications.length && <div className="empty-state"><div className="empty-icon"><Bell size={24}/></div><h3>All quiet on the movie front</h3><p>When a new movie matches your cluster, it will show up here.</p><button className="button button-outline" onClick={() => setShowNewMovie(true)}>Add a test movie</button></div>}
            {notifications.map((note) => <div className={`notification-card ${!note.read ? "unread" : ""}`} key={note.notification_id}><div className="notification-art"><Film size={22}/></div><div className="notification-body"><div className="notification-meta"><span>{!note.read ? "NEW DISCOVERY" : "DISCOVERY"} · {new Date(note.created_at).toLocaleString()}</span>{!note.read && <i/>}</div><p>{note.message}</p><small>Movie ID: {note.movie_id}</small></div>{!note.read && <button className="button button-outline button-small" onClick={() => markRead(note)}><Check size={14}/> Mark read</button>}</div>)}
          </div></section>}

          {activeTab === "catalogue" && <section className="subpage"><div className="subpage-heading"><div className="eyebrow eyebrow-muted"><Film size={14}/> YOUR MOVIE SHELF</div><h1>My catalogue</h1><p>Movies added to the application's local database.</p></div><div className="catalogue-toolbar"><div className="search-box"><Search size={16}/><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search local catalogue" onKeyDown={(e) => e.key === "Enter" && loadMovies(search)}/></div><button className="button button-outline" onClick={() => loadMovies(search)}><Search size={15}/> Search</button><button className="button button-primary" onClick={() => setShowNewMovie(true)}><Plus size={15}/> Add movie</button></div><div className="catalogue-list">{movies.map((movie, index) => <div className="catalogue-row" key={movie.movie_id}><div className={`catalogue-thumb ${palette[index % palette.length]}`}><Film size={20}/></div><div className="catalogue-movie-info"><strong>{movie.title}</strong><span>{movie.genres?.join(" · ") || "Genres unavailable"}</span></div><span className="catalogue-id">#{movie.movie_id}</span><span className="catalogue-type">{movie.is_catalogue_movie ? "MOVIELENS" : "NEW RELEASE"}</span></div>)}{!movies.length && <div className="empty-state"><div className="empty-icon"><Film size={24}/></div><h3>Your shelf is ready</h3><p>Search for a movie added to this app, or add a new release to begin.</p><button className="button button-primary" onClick={() => setShowNewMovie(true)}><Plus size={15}/> Add a movie</button></div>}</div></section>}

          {activeTab === "taste" && <section className="subpage"><div className="subpage-heading"><div className="eyebrow eyebrow-muted"><Sparkles size={14}/> YOUR PERSONAL ALGORITHM</div><h1>Taste profile</h1><p>Your genre preferences are converted into a feature vector and assigned to a K-Means cluster.</p></div><div className="taste-layout"><div className="taste-panel"><div className="panel-title"><div><h3>Genre affinity</h3><p>Adjust the sliders, then save to update your cluster.</p></div><span className="pill">{user ? `Cluster ${user.cluster_id}` : "Guest"}</span></div><div className="preference-list">{genres.map((genre) => <PreferenceSlider key={genre} genre={genre} value={preferences[genre] ?? 3} onChange={(value) => setPreferences((old) => ({...old, [genre]: value}))}/>)}</div><button className="button button-primary save-preferences" onClick={user ? savePreferences : createUser} disabled={busyAction === "preferences" || loading}>{busyAction === "preferences" ? <LoaderCircle className="spin" size={16}/> : <Check size={16}/>} {user ? "Save taste profile" : "Create my profile"}</button></div><div className="cluster-detail"><div className="cluster-detail-icon"><Layers3 size={22}/></div><span className="eyebrow eyebrow-muted">YOUR ASSIGNMENT</span><div className="cluster-detail-number">{user ? `#${user.cluster_id}` : "—"}</div><h3>{user ? "Cluster assigned" : "Not clustered yet"}</h3><p>{user ? "Your profile is mapped to the nearest learned K-Means centroid. Similar users help shape your recommendations." : "Create a profile to see which preference group best matches your taste."}</p><div className="detail-divider"/><div className="detail-row"><span>Algorithm</span><strong>K-Means</strong></div><div className="detail-row"><span>Feature space</span><strong>Genre ratings</strong></div><div className="detail-row"><span>Preference scale</span><strong>1 – 5</strong></div></div></div></section>}
        </div>
        <footer className="footer"><span>reelcluster <span className="footer-dot">✦</span> made for movie people</span><span>Powered by K-Means clustering <span className="footer-status"><i/> ML API</span></span></footer>
      </main>

      {showOnboarding && <Modal title={user ? "Tune your taste" : "Your taste, your cinema"} subtitle="A few quick signals help us place you in a preference cluster." onClose={() => setShowOnboarding(false)} icon={<WandSparkles size={19}/>}>
        <div className="onboarding-intro"><div className="onboarding-art"><Clapperboard size={25}/><Sparkles size={16}/></div><div><strong>{user ? "Fine-tune your profile" : "Let's find your movie people."}</strong><span>Rate each genre from 1 (not for me) to 5 (love it).</span></div></div>
        <label className="field-label">USER ID <span>Required</span></label><input className="text-input" value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="e.g. movie_fan_01"/>
        <div className="modal-pref-heading"><span>GENRE PREFERENCES</span><span>1 — 5</span></div>
        <div className="onboarding-genres">{genres.slice(0, 12).map((genre) => <PreferenceSlider key={genre} genre={genre} value={preferences[genre] ?? 3} onChange={(value) => setPreferences((old) => ({...old, [genre]: value}))}/>)}</div>
        <button className="button button-primary modal-submit" onClick={user ? savePreferences : createUser} disabled={loading || busyAction === "preferences"}>{loading ? <LoaderCircle size={17} className="spin"/> : <Sparkles size={16}/>} {user ? "Save my preferences" : "Find my cluster"} <ArrowRight size={16}/></button>
        {user && <button className="text-button" onClick={() => setShowOnboarding(false)}>Cancel</button>}
      </Modal>}

      {showNewMovie && <Modal title="Introduce a new movie" subtitle="Find the preference clusters most likely to enjoy this release." onClose={() => setShowNewMovie(false)} icon={<Film size={19}/>}>
        <div className="new-movie-note"><Sparkles size={16}/><span>The model compares the movie's genres with average cluster ratings, then creates in-app notifications for matching users.</span></div>
        <label className="field-label">MOVIE ID <span>Unique identifier</span></label><input className="text-input" value={movieForm.movie_id} onChange={(e) => setMovieForm({...movieForm, movie_id: e.target.value})} placeholder="e.g. new-2026-01"/>
        <label className="field-label">MOVIE TITLE</label><input className="text-input" value={movieForm.title} onChange={(e) => setMovieForm({...movieForm, title: e.target.value})} placeholder="The name of the movie"/>
        <label className="field-label">DESCRIPTION <span>Optional</span></label><textarea className="text-input textarea" value={movieForm.description} onChange={(e) => setMovieForm({...movieForm, description: e.target.value})} placeholder="A short synopsis..." rows="2"/>
        <label className="field-label">GENRES <span>Select all that apply</span></label><div className="genre-picker">{genres.map((genre) => <button key={genre} className={`genre-chip ${movieForm.genres.includes(genre) ? "selected" : ""}`} onClick={() => setMovieForm((old) => ({...old, genres: old.genres.includes(genre) ? old.genres.filter((g) => g !== genre) : [...old.genres, genre]}))}><span>{movieForm.genres.includes(genre) ? <Check size={12}/> : <Plus size={12}/>}</span>{genre}</button>)}</div>
        <div className="form-row"><div><label className="field-label">MIN. GENRE SCORE</label><select className="text-input" value={movieForm.min_cluster_score} onChange={(e) => setMovieForm({...movieForm, min_cluster_score: Number(e.target.value)})}><option value="2.5">2.5 / 5</option><option value="3">3.0 / 5</option><option value="3.5">3.5 / 5</option><option value="4">4.0 / 5</option><option value="4.5">4.5 / 5</option></select></div><div><label className="field-label">MAX CLUSTERS</label><select className="text-input" value={movieForm.max_clusters} onChange={(e) => setMovieForm({...movieForm, max_clusters: Number(e.target.value)})}><option value="1">1 cluster</option><option value="2">2 clusters</option><option value="3">3 clusters</option><option value="5">5 clusters</option><option value="8">8 clusters</option></select></div></div>
        <button className="button button-primary modal-submit" onClick={submitMovie} disabled={busyAction === "new-movie" || !movieForm.movie_id.trim() || !movieForm.title.trim() || !movieForm.genres.length}>{busyAction === "new-movie" ? <LoaderCircle size={17} className="spin"/> : <Send size={16}/>} Add movie & target clusters <ArrowRight size={16}/></button>
      </Modal>}

      {showSettings && <Modal title="API connection" subtitle="Connect this frontend to your running FastAPI server." onClose={() => setShowSettings(false)} icon={<Settings2 size={19}/>}>
        <label className="field-label">FASTAPI BASE URL</label><input className="text-input" value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="http://127.0.0.1:8000"/>
        <div className="settings-help"><span className="pulse"/><span>Default: <code>http://127.0.0.1:8000</code>. Start the backend before connecting.</span></div>
        <button className="button button-primary modal-submit" onClick={async () => {localStorage.setItem("reelcluster_api", apiBase); await connect(); setShowSettings(false);}} disabled={loading}>{loading ? <LoaderCircle size={16} className="spin"/> : <Activity size={16}/>} Test connection <ArrowRight size={16}/></button>
        {user && <button className="button button-outline modal-submit" onClick={() => {localStorage.removeItem("reelcluster_user");setUserId("");setUser(null);setRecommendations([]);setNotifications([]);setShowSettings(false);setShowOnboarding(true);notify("Signed out on this browser");}}>Switch user</button>}
      </Modal>}

      {toast && <div className="toast"><span><Check size={15}/></span>{toast}</div>}
    </div>
  );
}

function SlidersIcon() {
  return <Settings2 size={18}/>;
}

function PreferenceSlider({ genre, value, onChange }) {
  return <div className="preference-row"><span className="preference-genre">{genre}</span><input aria-label={`${genre} preference`} type="range" min="1" max="5" step="1" value={value} onChange={(e) => onChange(Number(e.target.value))} style={{"--range-progress": `${((value - 1) / 4) * 100}%`}}/><span className={`preference-value ${value >= 4 ? "high" : value <= 2 ? "low" : ""}`}>{value}</span></div>;
}

function MovieCard({ movie, index, palette, genreEmoji, busyAction, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const tones = ["#b7a0ff", "#83b5ff", "#ffad7a", "#9ad9b0", "#f2a7d3", "#e5cf85"];
  const genre = movie.genres?.[0] || "CINEMA";
  return <article className="movie-card">
    <div className={`movie-art ${palette[index % palette.length]}`}>
      <div className="movie-art-grain"/><span className="movie-art-index">{String(index + 1).padStart(2, "0")} / PICK</span><span className="movie-art-glyph">{genreEmoji(movie)}</span><div className="movie-art-lines"/><div className="movie-art-title">{movie.title}</div><div className="movie-art-bottom"><span>{movie.genres?.slice(0, 2).join(" / ") || "MOVIE NIGHT"}</span><span>RC·{String(movie.movie_id).slice(-4)}</span></div>
      <div className="match-badge"><Sparkles size={11}/> CLUSTER PICK</div>
    </div>
    <div className="movie-card-content"><div className="movie-card-meta"><span className="genre-label">{genre}</span><span className="rating"><Star size={13} fill="currentColor"/>{Number(movie.average_cluster_rating || 0).toFixed(1)} <small>({movie.rating_count_in_cluster || 0})</small></span></div><h3 title={movie.title}>{movie.title}</h3><div className="movie-tags">{(movie.genres || []).slice(0, 3).map((g) => <span key={g}>{g}</span>)}</div><div className="movie-card-actions"><button title="I like this" onClick={() => onAction(movie, "like")} disabled={busyAction === `${movie.movie_id}-like`}><ThumbsUp size={15}/><span>Like</span></button><button title="Not for me" onClick={() => onAction(movie, "dislike")} disabled={busyAction === `${movie.movie_id}-dislike`}><ThumbsDown size={15}/></button><button title="Save to watchlist" className="save-movie" onClick={() => onAction(movie, "watchlist")} disabled={busyAction === `${movie.movie_id}-watchlist`}><Heart size={15}/></button></div></div>
  </article>;
}

function Modal({ title, subtitle, icon, onClose, children }) {
  useEffect(() => {
    const onKey = (event) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}><div className="modal">
    <div className="modal-header"><div className="modal-icon">{icon}</div><button className="icon-button" onClick={onClose} aria-label="Close modal"><X size={18}/></button></div>
    <h2>{title}</h2><p className="modal-subtitle">{subtitle}</p><div className="modal-content">{children}</div>
  </div></div>;
}

createRoot(document.getElementById("root")).render(<App/>);
