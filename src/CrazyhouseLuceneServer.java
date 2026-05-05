import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * CrazyhouseLuceneServer
 * ----------------------
 * Lucene 9.x HTTP search server for Crazyhouse tactical positions.
 *
 * First run: indexes corpus_mates.jsonl into a Lucene index on disk (~5-10 min).
 * Subsequent runs: opens existing index instantly, serves HTTP on port 8983.
 *
 * Endpoints:
 *   GET /search?q=<tokens>&topk=10&field=text_all&exclude_id=<id>
 *       Returns JSON array of hits: [{id, score, site, ply, board_fen,
 *                                     pockets_white, pockets_black,
 *                                     mate_before, turn, text_dynamic}, ...]
 *
 *   GET /doc?id=<doc_id>
 *       Returns JSON object for a single document by id.
 *
 *   GET /status
 *       Returns {"status":"ok","docs":<n>}
 *
 * Build:
 *   javac -cp "lucene/*" CrazyhouseLuceneServer.java
 *
 * Run:
 *   java -cp ".;lucene/*" CrazyhouseLuceneServer
 *   (Linux/Mac: use : instead of ; in classpath)
 */
public class CrazyhouseLuceneServer {

    // ── Paths ────────────────────────────────────────────────────────────────
    static final String CORPUS_PATH = "../data/derived/corpus_mates.jsonl";
    static final String INDEX_PATH  = "../data/derived/lucene_index";
    static final int    PORT        = 8983;
    static final int    CHUNK_SIZE  = 10_000;

    // ── Lucene globals ───────────────────────────────────────────────────────
    static IndexReader   reader;
    static IndexSearcher searcher;
    static Analyzer      analyzer = new WhitespaceAnalyzer();

    // ── Main ─────────────────────────────────────────────────────────────────
    public static void main(String[] args) throws Exception {
        Path indexPath  = Paths.get(INDEX_PATH);
        Path corpusPath = Paths.get(CORPUS_PATH);

        // Build index if it doesn't exist yet
        if (!Files.exists(indexPath) || !DirectoryReader.indexExists(FSDirectory.open(indexPath))) {
            System.out.println("Index not found — building from " + corpusPath);
            buildIndex(corpusPath, indexPath);
        } else {
            System.out.println("Index found at " + indexPath);
        }

        // Open index for searching
        FSDirectory dir = FSDirectory.open(indexPath);
        reader   = DirectoryReader.open(dir);
        searcher = new IndexSearcher(reader);
        // Use BM25 similarity (default in Lucene 9)
        searcher.setSimilarity(new org.apache.lucene.search.similarities.BM25Similarity());

        System.out.println("Index opened: " + reader.numDocs() + " documents");

        // Start HTTP server
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/search", new SearchHandler());
        server.createContext("/doc",    new DocHandler());
        server.createContext("/status", new StatusHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("Server started on port " + PORT);
        System.out.println("Ready.");
    }

    // ── Index builder ────────────────────────────────────────────────────────
    static void buildIndex(Path corpusPath, Path indexPath) throws Exception {
        Files.createDirectories(indexPath);
        FSDirectory dir = FSDirectory.open(indexPath);

        IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE);
        cfg.setRAMBufferSizeMB(512);

        long t0 = System.currentTimeMillis();
        int  n  = 0;

        try (IndexWriter writer = new IndexWriter(dir, cfg);
             BufferedReader br   = Files.newBufferedReader(corpusPath, StandardCharsets.UTF_8)) {

            long lineCount = 0;
            String line;
            while ((line = br.readLine()) != null) {
                lineCount++;
                if (lineCount == 1) System.out.println("  First line length: " + line.length() + " chars");
                if (lineCount == 1) System.out.println("  First 120 chars: " + line.substring(0, Math.min(120, line.length())));

                line = line.trim();
                if (line.isEmpty()) continue;

                try {
                    Document doc = jsonLineToDocument(line);
                    if (doc != null) {
                        writer.addDocument(doc);
                        n++;
                        if (n % CHUNK_SIZE == 0) {
                            writer.flush();
                            long elapsed = (System.currentTimeMillis() - t0) / 1000;
                            System.out.printf("  indexed %,d docs  (%ds)%n", n, elapsed);
                        }
                    }
                } catch (Exception e) {
                    // skip malformed lines — print first failure for debug
                    if (n == 0) System.out.println("  Parse error on line: " + e.getMessage());
                }
            }

            System.out.println("Committing...");
            writer.commit();
        }

        long elapsed = (System.currentTimeMillis() - t0) / 1000;
        System.out.printf("Index built: %,d docs in %ds%n", n, elapsed);
    }

    // ── JSON line → Lucene Document ──────────────────────────────────────────
    static Document jsonLineToDocument(String line) {
        // Minimal JSON parser — extract known fields without a JSON library
        String id          = jsonStr(line, "id");
        if (id == null || id.isEmpty()) return null;

        String site        = jsonStr(line, "site");
        String boardFen    = jsonStr(line, "board_fen");
        String turn        = jsonStr(line, "turn");
        // pockets are JSON objects {}, extract as raw JSON blob
        String pwJson      = jsonObj(line, "pockets_white");
        String pbJson      = jsonObj(line, "pockets_black");
        String textAll     = jsonStr(line, "text_all");
        String textStatic  = jsonStr(line, "text_static");
        String textDynamic = jsonStr(line, "text_dynamic");
        String textDynGen  = jsonStr(line, "text_dynamic_general");
        String textDynSol  = jsonStr(line, "text_dynamic_solution");
        int    ply         = jsonInt(line, "ply");
        int    mate        = jsonInt(line, "mate_before");
        // Metadata fields (feature_descriptions_summary_2_.md section 4.3)
        int    metaLength  = jsonInt(line,    "meta_length");
        double metaDelta   = jsonDouble(line, "meta_delta");
        double metaCp      = jsonDouble(line, "meta_cp_before");
        int    metaMateIn  = jsonInt(line,    "meta_mate_in");

        Document doc = new Document();

        // Stored + indexed fields
        doc.add(new StringField("id",           id,                       Field.Store.YES));
        doc.add(new StoredField("site",         site        != null ? site        : ""));
        doc.add(new StoredField("board_fen",    boardFen    != null ? boardFen    : ""));
        doc.add(new StoredField("turn",         turn        != null ? turn        : "white"));
        doc.add(new StoredField("pockets_white",pwJson      != null ? pwJson      : "{}"));
        doc.add(new StoredField("pockets_black",pbJson      != null ? pbJson      : "{}"));
        doc.add(new StoredField("text_dynamic", textDynamic != null ? textDynamic : ""));
        doc.add(new IntPoint(   "ply",          ply));
        doc.add(new StoredField("ply",          ply));
        doc.add(new IntPoint(   "mate_before",  mate));
        doc.add(new StoredField("mate_before",  mate));
        // Metadata stored fields — returned by /search and /doc endpoints
        doc.add(new StoredField("meta_length",   metaLength));
        doc.add(new StoredField("meta_delta",    metaDelta));
        doc.add(new StoredField("meta_cp_before",metaCp));
        doc.add(new StoredField("meta_mate_in",  metaMateIn));

        // Full-text search fields (not stored to save space)
        doc.add(new TextField("text_all",              textAll     != null ? textAll     : "", Field.Store.NO));
        doc.add(new TextField("text_static",           textStatic  != null ? textStatic  : "", Field.Store.NO));
        doc.add(new TextField("text_dynamic_general",  textDynGen  != null ? textDynGen  : "", Field.Store.NO));
        doc.add(new TextField("text_dynamic_solution", textDynSol  != null ? textDynSol  : "", Field.Store.NO));

        return doc;
    }

    // ── Search handler ───────────────────────────────────────────────────────
    static class SearchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());
                String q         = params.getOrDefault("q", "");
                String field     = params.getOrDefault("field", "text_all");
                int    topk      = Integer.parseInt(params.getOrDefault("topk", "10"));
                String excludeId = params.getOrDefault("exclude_id", "");
                String excludeBase = baseGameId(excludeId);

                if (q.isEmpty()) {
                    respond(ex, 400, "{\"error\":\"q required\"}");
                    return;
                }

                // Escape special Lucene chars, parse as whitespace-split OR query
                BooleanQuery.Builder bqb = new BooleanQuery.Builder();
                for (String token : q.split("\\s+")) {
                    if (token.isEmpty()) continue;
                    // Escape special chars
                    String safe = QueryParser.escape(token);
                    bqb.add(new BoostQuery(new TermQuery(new Term(field, safe)), 1.0f),
                            BooleanClause.Occur.SHOULD);
                }
                bqb.setMinimumNumberShouldMatch(1);
                Query query = bqb.build();

                // Fetch more than needed to allow filtering
                int fetch = Math.min(topk * 10 + 50, reader.numDocs());
                TopDocs topDocs = searcher.search(query, fetch);

                StringBuilder sb = new StringBuilder("[");
                int rank = 0;
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    if (rank >= topk) break;
                    Document doc = searcher.storedFields().document(sd.doc);
                    String docId = doc.get("id");

                    // Filter: skip self and same-game docs
                    if (docId.equals(excludeId)) continue;
                    if (!excludeBase.isEmpty() && baseGameId(docId).equals(excludeBase)) continue;

                    if (rank > 0) sb.append(",");
                    sb.append(docToJson(doc, sd.score, rank + 1));
                    rank++;
                }
                sb.append("]");

                respond(ex, 200, sb.toString());

            } catch (Exception e) {
                respond(ex, 500, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
            }
        }
    }

    // ── Doc handler ──────────────────────────────────────────────────────────
    static class DocHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());
                String id = params.getOrDefault("id", "");
                if (id.isEmpty()) {
                    respond(ex, 400, "{\"error\":\"id required\"}");
                    return;
                }

                Query q = new TermQuery(new Term("id", id));
                TopDocs td = searcher.search(q, 1);
                if (td.totalHits.value == 0) {
                    respond(ex, 404, "{\"error\":\"not found\"}");
                    return;
                }

                Document doc = searcher.storedFields().document(td.scoreDocs[0].doc);
                respond(ex, 200, docToJson(doc, 1.0f, 0));

            } catch (Exception e) {
                respond(ex, 500, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
            }
        }
    }

    // ── Status handler ───────────────────────────────────────────────────────
    static class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            respond(ex, 200,
                "{\"status\":\"ok\",\"docs\":" + reader.numDocs() + "}");
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    static String docToJson(Document doc, float score, int rank) {
        return String.format(
            "{\"id\":%s,\"rank\":%d,\"score\":%.4f," +
            "\"site\":%s,\"ply\":%s,\"mate_before\":%s," +
            "\"board_fen\":%s,\"turn\":%s," +
            "\"pockets_white\":%s,\"pockets_black\":%s," +
            "\"text_dynamic\":%s," +
            "\"meta_length\":%s,\"meta_delta\":%s," +
            "\"meta_cp_before\":%s,\"meta_mate_in\":%s}",
            jsonQuote(doc.get("id")),
            rank,
            score,
            jsonQuote(doc.get("site")),
            doc.get("ply")         != null ? doc.get("ply")         : "0",
            doc.get("mate_before") != null ? doc.get("mate_before") : "null",
            jsonQuote(doc.get("board_fen")),
            jsonQuote(doc.get("turn")),
            rawOrEmpty(doc.get("pockets_white")),
            rawOrEmpty(doc.get("pockets_black")),
            jsonQuote(doc.get("text_dynamic")),
            doc.get("meta_length")    != null ? doc.get("meta_length")    : "0",
            doc.get("meta_delta")     != null ? doc.get("meta_delta")     : "0",
            doc.get("meta_cp_before") != null ? doc.get("meta_cp_before") : "0",
            doc.get("meta_mate_in")   != null ? doc.get("meta_mate_in")   : "0"
        );
    }

    static String rawOrEmpty(String s) {
        // pockets_white/black are already stored as JSON objects
        return (s != null && !s.isEmpty()) ? s : "{}";
    }

    static String jsonQuote(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\")
                       .replace("\"", "\\\"")
                       .replace("\n", "\\n")
                       .replace("\r", "\\r") + "\"";
    }

    static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    static String baseGameId(String id) {
        if (id == null || id.isEmpty()) return "";
        int idx = id.lastIndexOf('_');
        return idx > 0 ? id.substring(0, idx) : id;
    }

    static void respond(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new LinkedHashMap<>();
        if (query == null || query.isEmpty()) return map;
        for (String part : query.split("&")) {
            int eq = part.indexOf('=');
            if (eq > 0) {
                String k = urlDecode(part.substring(0, eq));
                String v = urlDecode(part.substring(eq + 1));
                map.put(k, v);
            }
        }
        return map;
    }

    static String urlDecode(String s) {
        try {
            return java.net.URLDecoder.decode(s, "UTF-8");
        } catch (Exception e) {
            return s;
        }
    }

    // ── Minimal JSON field extractors ────────────────────────────────────────
    static String jsonStr(String json, String key) {
        // Matches "key": "value" or "key":"value" — handles spaces after colon
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return null;
        start += search.length();
        // skip whitespace
        while (start < json.length() && json.charAt(start) == ' ') start++;
        // expect opening quote
        if (start >= json.length() || json.charAt(start) != '\"') return null;
        start++; // skip opening quote
        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escape) {
                sb.append(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                break;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    static int jsonInt(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return 0;
        start += search.length();
        // skip whitespace
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length()) return 0;
        char first = json.charAt(start);
        if (first == 'n') return 0; // null
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (Character.isDigit(c) || c == '-') sb.append(c);
            else break;
        }
        try { return Integer.parseInt(sb.toString()); }
        catch (NumberFormatException e) { return 0; }
    }

    /** Extract a JSON double/float value. Returns 0.0 if key not found or value is null. */
    static double jsonDouble(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return 0.0;
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length()) return 0.0;
        char first = json.charAt(start);
        if (first == 'n') return 0.0; // null
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (Character.isDigit(c) || c == '-' || c == '.' || c == 'e' || c == 'E' || c == '+') sb.append(c);
            else break;
        }
        try { return Double.parseDouble(sb.toString()); }
        catch (NumberFormatException e) { return 0.0; }
    }

    /** Extract a JSON object value {"key":{...}} as raw string including braces. */
    static String jsonObj(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return "{}";
        start += search.length();
        // skip whitespace
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length() || json.charAt(start) != '{') return "{}";
        int depth = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') { depth--; sb.append(c); if (depth == 0) break; continue; }
            sb.append(c);
        }
        String result = sb.toString();
        return result.isEmpty() ? "{}" : result;
    }
}