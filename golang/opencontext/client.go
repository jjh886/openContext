// Package opencontext provides a Go client for the openContext AI context
// management REST API.
//
// Usage:
//
//	client := opencontext.NewClient(opencontext.Options{UserID: "alice"})
//
//	// Augment messages before sending to an LLM
//	result, err := client.Augment(ctx, opencontext.AugmentRequest{
//	    Messages: []opencontext.Message{
//	        {Role: "user", Content: "Which Go libraries should I use?"},
//	    },
//	    SessionID: "session-1",
//	})
//
//	// Record the conversation afterwards
//	_, err = client.Record(ctx, opencontext.RecordRequest{
//	    SessionID: "session-1",
//	    Messages:  append(result.Messages, opencontext.Message{Role: "assistant", Content: "..."}),
//	})
package opencontext

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// MemoryType represents the category of a memory.
type MemoryType string

const (
	MemoryTypeFact       MemoryType = "fact"
	MemoryTypeSkill      MemoryType = "skill"
	MemoryTypePreference MemoryType = "preference"
	MemoryTypeContact    MemoryType = "contact"
	MemoryTypeTask       MemoryType = "task"
	MemoryTypeSummary    MemoryType = "summary"
	MemoryTypeHabit      MemoryType = "habit"
)

// Message is a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Memory is a stored user memory.
type Memory struct {
	ID              string                 `json:"id"`
	Content         string                 `json:"content"`
	MemoryType      MemoryType             `json:"memory_type"`
	Importance      float64                `json:"importance"`
	Tags            []string               `json:"tags"`
	SourceSessionID *string                `json:"source_session_id"`
	CreatedAt       string                 `json:"created_at"`
	UpdatedAt       string                 `json:"updated_at"`
	LastAccessed    string                 `json:"last_accessed"`
	AccessCount     int                    `json:"access_count"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// UserProfile is the aggregated profile derived from user memories.
type UserProfile struct {
	UserID      string                 `json:"user_id"`
	Name        *string                `json:"name"`
	Skills      []string               `json:"skills"`
	Preferences map[string]interface{} `json:"preferences"`
	Habits      []string               `json:"habits"`
	Facts       []string               `json:"facts"`
	UpdatedAt   string                 `json:"updated_at"`
	Exists      bool                   `json:"exists"`
}

// Session is a conversation session summary.
type Session struct {
	ID           string `json:"id"`
	UserID       string `json:"user_id"`
	MessageCount int    `json:"message_count"`
	CreatedAt    string `json:"created_at"`
	UpdatedAt    string `json:"updated_at"`
}

// ContextResult holds the result of a context lookup.
type ContextResult struct {
	InjectedSystemPrompt string             `json:"injected_system_prompt"`
	Memories             []Memory           `json:"memories"`
	RelevanceScores      map[string]float64 `json:"relevance_scores"`
}

// AugmentRequest is the payload for /api/v1/augment.
type AugmentRequest struct {
	Messages  []Message `json:"messages"`
	SessionID string    `json:"session_id,omitempty"`
}

// AugmentResult is the response from /api/v1/augment.
type AugmentResult struct {
	Messages []Message `json:"messages"`
}

// RecordRequest is the payload for /api/v1/record.
type RecordRequest struct {
	SessionID string    `json:"session_id"`
	Messages  []Message `json:"messages"`
}

// RecordResult is the response from /api/v1/record.
type RecordResult struct {
	SessionID    string `json:"session_id"`
	MessageCount int    `json:"message_count"`
}

// AddMemoryRequest is the payload for POST /api/v1/memories.
type AddMemoryRequest struct {
	Content    string     `json:"content"`
	MemoryType MemoryType `json:"memory_type"`
	Importance float64    `json:"importance"`
	Tags       []string   `json:"tags"`
}

// MemoryListResult is the response from GET /api/v1/memories.
type MemoryListResult struct {
	Memories []Memory `json:"memories"`
	Count    int      `json:"count"`
}

// SessionListResult is the response from GET /api/v1/sessions.
type SessionListResult struct {
	Sessions []Session `json:"sessions"`
	Count    int       `json:"count"`
}

// Options configure the client.
type Options struct {
	// UserID is the default user for all requests.
	UserID string
	// ServerURL is the base URL of the openContext server. Default: http://localhost:8765
	ServerURL string
	// Timeout for HTTP requests. Default: 10s
	Timeout time.Duration
}

// Client is the openContext API client.
type Client struct {
	userID    string
	serverURL string
	http      *http.Client
}

// NewClient creates a new openContext client.
func NewClient(opts Options) *Client {
	serverURL := opts.ServerURL
	if serverURL == "" {
		serverURL = "http://localhost:8765"
	}
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	return &Client{
		userID:    opts.UserID,
		serverURL: serverURL,
		http:      &http.Client{Timeout: timeout},
	}
}

// Augment injects relevant user memories into messages before sending to an LLM.
func (c *Client) Augment(ctx context.Context, req AugmentRequest) (*AugmentResult, error) {
	type body struct {
		UserID    string    `json:"user_id"`
		Messages  []Message `json:"messages"`
		SessionID string    `json:"session_id,omitempty"`
	}
	var result AugmentResult
	err := c.post(ctx, "/api/v1/augment", body{
		UserID:    c.userID,
		Messages:  req.Messages,
		SessionID: req.SessionID,
	}, &result)
	return &result, err
}

// Record persists a conversation and extracts new memories.
func (c *Client) Record(ctx context.Context, req RecordRequest) (*RecordResult, error) {
	type body struct {
		UserID    string    `json:"user_id"`
		SessionID string    `json:"session_id"`
		Messages  []Message `json:"messages"`
	}
	var result RecordResult
	err := c.post(ctx, "/api/v1/record", body{
		UserID:    c.userID,
		SessionID: req.SessionID,
		Messages:  req.Messages,
	}, &result)
	return &result, err
}

// GetContext retrieves relevant context for a query.
func (c *Client) GetContext(ctx context.Context, query string, sessionID string) (*ContextResult, error) {
	params := url.Values{"user_id": {c.userID}, "query": {query}}
	if sessionID != "" {
		params.Set("session_id", sessionID)
	}
	var result ContextResult
	err := c.get(ctx, "/api/v1/context?"+params.Encode(), &result)
	return &result, err
}

// AddMemory manually adds a memory for the current user.
func (c *Client) AddMemory(ctx context.Context, req AddMemoryRequest) (*Memory, error) {
	type body struct {
		UserID     string     `json:"user_id"`
		Content    string     `json:"content"`
		MemoryType MemoryType `json:"memory_type"`
		Importance float64    `json:"importance"`
		Tags       []string   `json:"tags"`
	}
	if req.MemoryType == "" {
		req.MemoryType = MemoryTypeFact
	}
	if req.Importance == 0 {
		req.Importance = 0.7
	}
	if req.Tags == nil {
		req.Tags = []string{}
	}
	var result Memory
	err := c.post(ctx, "/api/v1/memories", body{
		UserID:     c.userID,
		Content:    req.Content,
		MemoryType: req.MemoryType,
		Importance: req.Importance,
		Tags:       req.Tags,
	}, &result)
	return &result, err
}

// GetMemories lists memories for the current user.
func (c *Client) GetMemories(ctx context.Context, memoryType MemoryType, limit int) (*MemoryListResult, error) {
	params := url.Values{"user_id": {c.userID}}
	if memoryType != "" {
		params.Set("memory_type", string(memoryType))
	}
	if limit > 0 {
		params.Set("limit", fmt.Sprintf("%d", limit))
	}
	var result MemoryListResult
	err := c.get(ctx, "/api/v1/memories?"+params.Encode(), &result)
	return &result, err
}

// DeleteMemory deletes a memory by ID.
func (c *Client) DeleteMemory(ctx context.Context, memoryID string) error {
	return c.delete(ctx, "/api/v1/memories/"+url.PathEscape(memoryID))
}

// SearchMemories performs a full-text search over user memories.
func (c *Client) SearchMemories(ctx context.Context, query string, limit int) (*MemoryListResult, error) {
	params := url.Values{"user_id": {c.userID}, "query": {query}}
	if limit > 0 {
		params.Set("limit", fmt.Sprintf("%d", limit))
	}
	var result MemoryListResult
	err := c.get(ctx, "/api/v1/memories/search?"+params.Encode(), &result)
	return &result, err
}

// GetUserProfile returns the aggregated user profile.
func (c *Client) GetUserProfile(ctx context.Context) (*UserProfile, error) {
	var result UserProfile
	err := c.get(ctx, "/api/v1/profile?user_id="+url.QueryEscape(c.userID), &result)
	return &result, err
}

// GetSessions lists recent sessions for the current user.
func (c *Client) GetSessions(ctx context.Context, limit int) (*SessionListResult, error) {
	params := url.Values{"user_id": {c.userID}}
	if limit > 0 {
		params.Set("limit", fmt.Sprintf("%d", limit))
	}
	var result SessionListResult
	err := c.get(ctx, "/api/v1/sessions?"+params.Encode(), &result)
	return &result, err
}

// ClearUserData deletes all data for the current user (GDPR / privacy).
func (c *Client) ClearUserData(ctx context.Context) error {
	return c.delete(ctx, "/api/v1/users/"+url.PathEscape(c.userID))
}

// ------------------------------------------------------------------ //
// HTTP helpers
// ------------------------------------------------------------------ //

func (c *Client) post(ctx context.Context, path string, payload interface{}, out interface{}) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("opencontext: marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.serverURL+path, bytes.NewReader(b))
	if err != nil {
		return fmt.Errorf("opencontext: new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return c.do(req, out)
}

func (c *Client) get(ctx context.Context, path string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.serverURL+path, nil)
	if err != nil {
		return fmt.Errorf("opencontext: new request: %w", err)
	}
	return c.do(req, out)
}

func (c *Client) delete(ctx context.Context, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.serverURL+path, nil)
	if err != nil {
		return fmt.Errorf("opencontext: new request: %w", err)
	}
	return c.do(req, nil)
}

func (c *Client) do(req *http.Request, out interface{}) error {
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("opencontext: http: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("opencontext: server returned %d: %s", resp.StatusCode, string(body))
	}
	if out != nil && len(body) > 0 {
		if err := json.Unmarshal(body, out); err != nil {
			return fmt.Errorf("opencontext: decode: %w", err)
		}
	}
	return nil
}
