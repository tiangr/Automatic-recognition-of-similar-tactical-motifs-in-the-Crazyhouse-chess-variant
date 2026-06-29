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
 * First run: indexes corpus_mates.jsonl into a Lucene index on disk.
 * Subsequent runs: opens existing index instantly, serves HTTP on port 8983.
 *
 * Endpoints:
 *   GET /search?q=<tokens>&topk=10&field=text_all&exclude_id=<id>
 *   GET /doc?id=<doc_id>
 *   GET /status
 *
 * Build:
 *   javac -cp "lucene/*" CrazyhouseLuceneServer.java
 *
 * Run:
 *   java -cp ".;lucene/*" CrazyhouseLuceneServer
 *   (Linux/Mac: use : instead of ; in classpath)
 *
 * To rebuild index: delete ../data/derived/lucene_index/ and restart.
 */
public class CrazyhouseLuceneServer {

    static final String CORPUS_PATH = "../data/derived/corpus_checkmates5.jsonl";
    static final String INDEX_PATH  = "../data/derived/lucene_index";
    static final int    PORT        = 8983;
    static final int    CHUNK_SIZE  = 10_000;

    static IndexReader   reader;
    static IndexSearcher searcher;
    static Analyzer      analyzer = new WhitespaceAnalyzer();

    public static void main(String[] args) throws Exception {
        Path indexPath  = Paths.get(INDEX_PATH);
        Path corpusPath = Paths.get(CORPUS_PATH);

        if (!Files.exists(indexPath) || !DirectoryReader.indexExists(FSDirectory.open(indexPath))) {
            System.out.println("Index not found — building from " + corpusPath);
            buildIndex(corpusPath, indexPath);
        } else {
            System.out.println("Index found at " + indexPath);
        }

        FSDirectory dir = FSDirectory.open(indexPath);
        reader   = DirectoryReader.open(dir);
        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new org.apache.lucene.search.similarities.BM25Similarity());

        System.out.println("Index opened: " + reader.numDocs() + " documents");

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/search", new SearchHandler());
        server.createContext("/rrf",    new RrfHandler());
        server.createContext("/doc",    new DocHandler());
        server.createContext("/status", new StatusHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("Server started on port " + PORT);
        System.out.println("Ready.");
    }

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

            String line;
            while ((line = br.readLine()) != null) {
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
                    if (n == 0) System.out.println("  Parse error: " + e.getMessage());
                }
            }
            System.out.println("Committing...");
            writer.commit();
        }

        long elapsed = (System.currentTimeMillis() - t0) / 1000;
        System.out.printf("Index built: %,d docs in %ds%n", n, elapsed);
    }

    static Document jsonLineToDocument(String line) {
        String id = jsonStr(line, "id");
        if (id == null || id.isEmpty()) return null;

        Document doc = new Document();

        // Identity
        doc.add(new StringField("id",  id, Field.Store.YES));
        doc.add(new StoredField("site", strOrEmpty(jsonStr(line, "site"))));
        doc.add(new IntPoint(  "ply",   jsonInt(line, "ply")));
        doc.add(new StoredField("ply",  jsonInt(line, "ply")));

        // Board state
        doc.add(new StoredField("board_fen",     strOrEmpty(jsonStr(line, "board_fen"))));
        doc.add(new StoredField("fen",           strOrEmpty(jsonStr(line, "fen"))));
        doc.add(new StoredField("turn",          strOrEmpty(jsonStr(line, "turn"))));
        doc.add(new StoredField("pockets_white", strOrEmpty(jsonObj(line, "pockets_white"))));
        doc.add(new StoredField("pockets_black", strOrEmpty(jsonObj(line, "pockets_black"))));

        // Puzzle
        doc.add(new StoredField("mate_in",        jsonInt(line, "mate_in")));
        doc.add(new StoredField("mate_before",     jsonInt(line, "mate_before")));  // backwards compat
        doc.add(new StoredField("solution_uci",    strOrEmpty(jsonArr(line, "solution_uci"))));
        doc.add(new StoredField("solution_san",    strOrEmpty(jsonArr(line, "solution_san"))));
        doc.add(new StoredField("engine_verified", jsonBool(line, "engine_verified") ? "true" : "false"));

        // Game metadata
        doc.add(new StoredField("white",           strOrEmpty(jsonStr(line, "white"))));
        doc.add(new StoredField("black",           strOrEmpty(jsonStr(line, "black"))));
        doc.add(new StoredField("white_elo",       jsonInt(line, "white_elo")));
        doc.add(new StoredField("black_elo",       jsonInt(line, "black_elo")));
        doc.add(new StoredField("game_rating_avg", jsonInt(line, "game_rating_avg")));
        doc.add(new StoredField("time_control",    strOrEmpty(jsonStr(line, "time_control"))));
        doc.add(new StoredField("utc_date",        strOrEmpty(jsonStr(line, "utc_date"))));
        doc.add(new StoredField("result",          strOrEmpty(jsonStr(line, "result"))));
        doc.add(new StoredField("source_pgn",      strOrEmpty(jsonStr(line, "source_pgn"))));
        doc.add(new StoredField("event",           strOrEmpty(jsonStr(line, "event"))));

        // Meta
        doc.add(new StoredField("meta_length",     jsonInt(line, "meta_length")));
        doc.add(new StoredField("meta_mate_in",    jsonInt(line, "meta_mate_in")));
        doc.add(new StoredField("meta_avg_rating", jsonInt(line, "meta_avg_rating")));

        // Text search fields
        String textAll    = strOrEmpty(jsonStr(line, "text_all"));
        String textStatic = strOrEmpty(jsonStr(line, "text_static"));
        String textDyn    = strOrEmpty(jsonStr(line, "text_dynamic"));
        String textDynGen = strOrEmpty(jsonStr(line, "text_dynamic_general"));
        String textDynSol = strOrEmpty(jsonStr(line, "text_dynamic_solution"));
        String textMotif  = strOrEmpty(jsonStr(line, "text_motif"));

        doc.add(new StoredField("text_dynamic", textDyn));
        doc.add(new StoredField("text_static",  textStatic));

        doc.add(new TextField("text_all",              textAll,    Field.Store.NO));
        doc.add(new TextField("text_static",           textStatic, Field.Store.NO));
        doc.add(new TextField("text_dynamic_general",  textDynGen, Field.Store.NO));
        doc.add(new TextField("text_dynamic_solution", textDynSol, Field.Store.NO));
        doc.add(new TextField("text_motif",            textMotif,  Field.Store.NO));

        return doc;
    }

    // ── Handlers ─────────────────────────────────────────────────────────────

    static class SearchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());
                String q           = params.getOrDefault("q", "");
                String field       = params.getOrDefault("field", "text_all");
                int    topk        = Integer.parseInt(params.getOrDefault("topk", "10"));
                String excludeId   = params.getOrDefault("exclude_id", "");
                String excludeBase = baseGameId(excludeId);

                if (q.isEmpty()) { respond(ex, 400, "{\"error\":\"q required\"}"); return; }

                BooleanQuery.Builder bqb = new BooleanQuery.Builder();
                for (String token : q.split("\\s+")) {
                    if (token.isEmpty()) continue;
                    bqb.add(new BoostQuery(new TermQuery(new Term(field, token)), 1.0f),
                            BooleanClause.Occur.SHOULD);
                }
                bqb.setMinimumNumberShouldMatch(1);

                int fetch = Math.min(topk * 10 + 50, reader.numDocs());
                TopDocs topDocs = searcher.search(bqb.build(), fetch);

                StringBuilder sb = new StringBuilder("[");
                int rank = 0;
                for (ScoreDoc sd : topDocs.scoreDocs) {
                    if (rank >= topk) break;
                    Document doc = searcher.storedFields().document(sd.doc);
                    String docId = doc.get("id");
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

    /**
     * Reciprocal Rank Fusion across multiple fields. Each field is queried
     * independently (per-field BM25 ranking); a document's fused score is
     * sum over fields of weight_field / (K + rank_in_that_field). A doc ranked
     * highly on ONE field (e.g. shares the mating-picture motif) surfaces even
     * if it ranks poorly on placement -- the "same motif, different board" case.
     *
     * GET /rrf?q=<tokens>&topk=10
     *         &fields=text_motif,text_static,text_dynamic_general,text_dynamic_solution
     *         &weights=2,1,1,1&k=60&exclude_id=<id>
     * The SAME q tokens are searched in every field (whitespace analyzer, so a
     * token that doesn't occur in a field simply doesn't match there).
     */
    static class RrfHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());
                String q           = params.getOrDefault("q", "");
                int    topk        = Integer.parseInt(params.getOrDefault("topk", "10"));
                int    K           = Integer.parseInt(params.getOrDefault("k", "60"));
                String excludeId   = params.getOrDefault("exclude_id", "");
                String excludeBase = baseGameId(excludeId);
                if (q.isEmpty()) { respond(ex, 400, "{\"error\":\"q required\"}"); return; }

                String[] fields = params.getOrDefault("fields",
                        "text_motif,text_static,text_dynamic_general,text_dynamic_solution")
                        .split(",");
                String[] wStr = params.getOrDefault("weights", "").split(",");
                double[] weights = new double[fields.length];
                for (int i = 0; i < fields.length; i++) {
                    weights[i] = 1.0;
                    if (i < wStr.length && !wStr[i].isEmpty()) {
                        try { weights[i] = Double.parseDouble(wStr[i]); } catch (Exception ignore) {}
                    }
                }

                String[] tokens = q.split("\\s+");
                int perField = Math.min(topk * 20 + 100, reader.numDocs());

                Map<Integer, Double> fused = new HashMap<>();
                for (int fi = 0; fi < fields.length; fi++) {
                    String field = fields[fi].trim();
                    if (field.isEmpty()) continue;
                    BooleanQuery.Builder bqb = new BooleanQuery.Builder();
                    boolean any = false;
                    for (String token : tokens) {
                        if (token.isEmpty()) continue;
                        bqb.add(new BoostQuery(new TermQuery(new Term(field, token)), 1.0f),
                                BooleanClause.Occur.SHOULD);
                        any = true;
                    }
                    if (!any) continue;
                    bqb.setMinimumNumberShouldMatch(1);
                    TopDocs td = searcher.search(bqb.build(), perField);
                    for (int rank = 0; rank < td.scoreDocs.length; rank++) {
                        int docId = td.scoreDocs[rank].doc;
                        double contrib = weights[fi] / (K + rank + 1.0);
                        fused.merge(docId, contrib, Double::sum);
                    }
                }

                List<Map.Entry<Integer, Double>> ordered = new ArrayList<>(fused.entrySet());
                ordered.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

                StringBuilder sb = new StringBuilder("[");
                int rank = 0;
                for (Map.Entry<Integer, Double> e : ordered) {
                    if (rank >= topk) break;
                    Document doc = searcher.storedFields().document(e.getKey());
                    String docId = doc.get("id");
                    if (docId.equals(excludeId)) continue;
                    if (!excludeBase.isEmpty() && baseGameId(docId).equals(excludeBase)) continue;
                    if (rank > 0) sb.append(",");
                    sb.append(docToJson(doc, e.getValue().floatValue(), rank + 1));
                    rank++;
                }
                sb.append("]");
                respond(ex, 200, sb.toString());

            } catch (Exception e) {
                respond(ex, 500, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
            }
        }
    }

    static class DocHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            try {
                Map<String, String> params = parseQuery(ex.getRequestURI().getQuery());
                String id = params.getOrDefault("id", "");
                if (id.isEmpty()) { respond(ex, 400, "{\"error\":\"id required\"}"); return; }

                TopDocs td = searcher.search(new TermQuery(new Term("id", id)), 1);
                if (td.totalHits.value == 0) { respond(ex, 404, "{\"error\":\"not found\"}"); return; }

                Document doc = searcher.storedFields().document(td.scoreDocs[0].doc);
                respond(ex, 200, docToJson(doc, 1.0f, 0));

            } catch (Exception e) {
                respond(ex, 500, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
            }
        }
    }

    static class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            respond(ex, 200, "{\"status\":\"ok\",\"docs\":" + reader.numDocs() + "}");
        }
    }

    // ── Document → JSON ──────────────────────────────────────────────────────

    static String docToJson(Document doc, float score, int rank) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"id\":")              .append(jsonQuote(doc.get("id")));
        sb.append(",\"rank\":")           .append(rank);
        sb.append(",\"score\":")          .append(String.format("%.4f", score));
        // Board
        sb.append(",\"site\":")           .append(jsonQuote(doc.get("site")));
        sb.append(",\"ply\":")            .append(intOrZero(doc.get("ply")));
        sb.append(",\"board_fen\":")      .append(jsonQuote(doc.get("board_fen")));
        sb.append(",\"fen\":")            .append(jsonQuote(doc.get("fen")));
        sb.append(",\"turn\":")           .append(jsonQuote(doc.get("turn")));
        sb.append(",\"pockets_white\":")  .append(rawOrObj(doc.get("pockets_white")));
        sb.append(",\"pockets_black\":")  .append(rawOrObj(doc.get("pockets_black")));
        // Puzzle
        sb.append(",\"mate_in\":")        .append(intOrNull(doc.get("mate_in")));
        sb.append(",\"mate_before\":")    .append(intOrNull(doc.get("mate_before")));
        sb.append(",\"solution_uci\":")   .append(arrOrEmpty(doc.get("solution_uci")));
        sb.append(",\"solution_san\":")   .append(arrOrEmpty(doc.get("solution_san")));
        sb.append(",\"engine_verified\":").append("true".equals(doc.get("engine_verified")) ? "true" : "false");
        // Game metadata
        sb.append(",\"white\":")          .append(jsonQuote(doc.get("white")));
        sb.append(",\"black\":")          .append(jsonQuote(doc.get("black")));
        sb.append(",\"white_elo\":")      .append(intOrNull(doc.get("white_elo")));
        sb.append(",\"black_elo\":")      .append(intOrNull(doc.get("black_elo")));
        sb.append(",\"game_rating_avg\":").append(intOrNull(doc.get("game_rating_avg")));
        sb.append(",\"time_control\":")   .append(jsonQuote(doc.get("time_control")));
        sb.append(",\"utc_date\":")       .append(jsonQuote(doc.get("utc_date")));
        sb.append(",\"result\":")         .append(jsonQuote(doc.get("result")));
        sb.append(",\"source_pgn\":")     .append(jsonQuote(doc.get("source_pgn")));
        sb.append(",\"event\":")          .append(jsonQuote(doc.get("event")));
        // Text
        sb.append(",\"text_dynamic\":")   .append(jsonQuote(doc.get("text_dynamic")));
        sb.append(",\"text_static\":")    .append(jsonQuote(doc.get("text_static")));
        // Meta
        sb.append(",\"meta_length\":")    .append(intOrZero(doc.get("meta_length")));
        sb.append(",\"meta_mate_in\":")   .append(intOrNull(doc.get("meta_mate_in")));
        sb.append(",\"meta_avg_rating\":").append(intOrNull(doc.get("meta_avg_rating")));
        sb.append("}");
        return sb.toString();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    static String strOrEmpty(String s)   { return s != null ? s : ""; }
    static String rawOrObj(String s)     { return (s != null && !s.isEmpty()) ? s : "{}"; }
    static String arrOrEmpty(String s)   { return (s != null && !s.isEmpty()) ? s : "[]"; }
    static String intOrZero(String s)    { try { Integer.parseInt(s); return s; } catch (Exception e) { return "0"; } }
    static String intOrNull(String s)    {
        if (s == null || s.isEmpty() || s.equals("0")) return "null";
        try { Integer.parseInt(s); return s; } catch (Exception e) { return "null"; }
    }

    static String jsonQuote(String s) {
        if (s == null || s.isEmpty()) return "null";
        return "\"" + s.replace("\\","\\\\").replace("\"","\\\"")
                       .replace("\n","\\n").replace("\r","\\r") + "\"";
    }

    static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\","\\\\").replace("\"","\\\"");
    }

    static String baseGameId(String id) {
        if (id == null || id.isEmpty()) return "";
        int idx = id.lastIndexOf('_');
        return idx > 0 ? id.substring(0, idx) : id;
    }

    static void respond(HttpExchange ex, int code, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.sendResponseHeaders(code, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new LinkedHashMap<>();
        if (query == null || query.isEmpty()) return map;
        for (String part : query.split("&")) {
            int eq = part.indexOf('=');
            if (eq > 0) map.put(urlDecode(part.substring(0, eq)), urlDecode(part.substring(eq + 1)));
        }
        return map;
    }

    static String urlDecode(String s) {
        try { return java.net.URLDecoder.decode(s, "UTF-8"); } catch (Exception e) { return s; }
    }

    // ── JSON extractors ──────────────────────────────────────────────────────

    static String jsonStr(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return null;
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length() || json.charAt(start) != '"') return null;
        start++;
        StringBuilder sb = new StringBuilder();
        boolean escape = false;
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escape) { sb.append(c); escape = false; }
            else if (c == '\\') { escape = true; }
            else if (c == '"') { break; }
            else { sb.append(c); }
        }
        return sb.toString();
    }

    static int jsonInt(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return 0;
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length() || json.charAt(start) == 'n') return 0;
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (Character.isDigit(c) || c == '-') sb.append(c);
            else break;
        }
        try { return Integer.parseInt(sb.toString()); } catch (NumberFormatException e) { return 0; }
    }

    static boolean jsonBool(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return false;
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        return start < json.length() && json.charAt(start) == 't';
    }

    static String jsonArr(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return "[]";
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length() || json.charAt(start) != '[') return "[]";
        int depth = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '[') depth++;
            else if (c == ']') { sb.append(c); if (--depth == 0) break; continue; }
            sb.append(c);
        }
        return sb.length() > 0 ? sb.toString() : "[]";
    }

    static String jsonObj(String json, String key) {
        String search = "\"" + key + "\":";
        int start = json.indexOf(search);
        if (start < 0) return "{}";
        start += search.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        if (start >= json.length() || json.charAt(start) != '{') return "{}";
        int depth = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') { sb.append(c); if (--depth == 0) break; continue; }
            sb.append(c);
        }
        return sb.length() > 0 ? sb.toString() : "{}";
    }
}