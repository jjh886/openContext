package opencontext_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	opencontext "github.com/jjh886/openContext/golang/opencontext"
)

// ------------------------------------------------------------------ //
// Helpers
// ------------------------------------------------------------------ //

func newTestServer(handler http.HandlerFunc) (*httptest.Server, *opencontext.Client) {
	srv := httptest.NewServer(handler)
	client := opencontext.NewClient(opencontext.Options{
		UserID:    "test-user",
		ServerURL: srv.URL,
	})
	return srv, client
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

// ------------------------------------------------------------------ //
// Tests
// ------------------------------------------------------------------ //

func TestAugment(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/v1/augment" {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		var req struct {
			UserID   string              `json:"user_id"`
			Messages []opencontext.Message `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.UserID != "test-user" {
			http.Error(w, "wrong user_id", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]interface{}{
			"messages": []map[string]string{
				{"role": "system", "content": "## Context"},
				{"role": "user", "content": "Hello"},
			},
		})
	})
	defer srv.Close()

	result, err := client.Augment(context.Background(), opencontext.AugmentRequest{
		Messages: []opencontext.Message{{Role: "user", Content: "Hello"}},
	})
	if err != nil {
		t.Fatalf("Augment error: %v", err)
	}
	if len(result.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(result.Messages))
	}
	if result.Messages[0].Role != "system" {
		t.Errorf("expected first message to be system, got %q", result.Messages[0].Role)
	}
}

func TestRecord(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/v1/record" {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]interface{}{
			"session_id":    "sess-1",
			"message_count": 2,
		})
	})
	defer srv.Close()

	result, err := client.Record(context.Background(), opencontext.RecordRequest{
		SessionID: "sess-1",
		Messages: []opencontext.Message{
			{Role: "user", Content: "I use Go"},
			{Role: "assistant", Content: "Great!"},
		},
	})
	if err != nil {
		t.Fatalf("Record error: %v", err)
	}
	if result.SessionID != "sess-1" {
		t.Errorf("expected session-id sess-1, got %q", result.SessionID)
	}
	if result.MessageCount != 2 {
		t.Errorf("expected message_count 2, got %d", result.MessageCount)
	}
}

func TestAddMemory(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/v1/memories" {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		var req struct {
			UserID  string `json:"user_id"`
			Content string `json:"content"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		writeJSON(w, map[string]interface{}{
			"id":               "mem-1",
			"content":          req.Content,
			"memory_type":      "fact",
			"importance":       0.7,
			"tags":             []string{},
			"source_session_id": nil,
			"created_at":       "2024-01-01T00:00:00",
			"updated_at":       "2024-01-01T00:00:00",
			"last_accessed":    "2024-01-01T00:00:00",
			"access_count":     0,
			"metadata":         map[string]interface{}{},
		})
	})
	defer srv.Close()

	mem, err := client.AddMemory(context.Background(), opencontext.AddMemoryRequest{
		Content: "I live in Berlin",
	})
	if err != nil {
		t.Fatalf("AddMemory error: %v", err)
	}
	if mem.ID != "mem-1" {
		t.Errorf("expected memory id mem-1, got %q", mem.ID)
	}
}

func TestGetMemories(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/api/v1/memories" {
			http.Error(w, "unexpected", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]interface{}{
			"memories": []interface{}{},
			"count":    0,
		})
	})
	defer srv.Close()

	result, err := client.GetMemories(context.Background(), "", 0)
	if err != nil {
		t.Fatalf("GetMemories error: %v", err)
	}
	if result.Count != 0 {
		t.Errorf("expected count 0, got %d", result.Count)
	}
}

func TestClearUserData(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "expected DELETE", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]interface{}{"cleared": true, "user_id": "test-user"})
	})
	defer srv.Close()

	if err := client.ClearUserData(context.Background()); err != nil {
		t.Fatalf("ClearUserData error: %v", err)
	}
}

func TestServerError(t *testing.T) {
	srv, client := newTestServer(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	})
	defer srv.Close()

	_, err := client.GetUserProfile(context.Background())
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
