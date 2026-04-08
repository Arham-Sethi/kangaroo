/**
 * Simple content hasher for message deduplication.
 *
 * Uses a fast non-crypto hash (djb2) since we only need
 * dedup identity, not security.
 */

export function hashMessage(role: string, content: string, index: number): string {
  const input = `${role}:${index}:${content.slice(0, 200)}`;
  return djb2(input);
}

function djb2(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) | 0;
  }
  return (hash >>> 0).toString(36);
}

export function generateConversationId(platform: string, url: string): string {
  const cleaned = url.replace(/[?#].*$/, "");
  return `${platform}:${djb2(cleaned)}`;
}
