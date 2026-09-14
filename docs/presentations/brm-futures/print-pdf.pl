#!/usr/bin/env perl

use strict;
use warnings;
use IO::Select;
use IO::Socket::INET;
use JSON::PP qw(decode_json encode_json);
use MIME::Base64 qw(decode_base64 encode_base64);
use POSIX qw(WNOHANG);
use Time::HiRes qw(sleep time);

my ($chrome, $html_path, $pdf_path, $expected_pages, $profile) = @ARGV;
defined $profile
  or die "usage: $0 CHROME HTML PDF EXPECTED_PAGES PROFILE\n";
$expected_pages =~ /^\d+$/ && $expected_pages > 0
  or die "EXPECTED_PAGES must be a positive integer\n";

sub read_exact {
  my ($socket, $length) = @_;
  my $select = IO::Select->new($socket);
  my $buffer = '';
  while (length($buffer) < $length) {
    $select->can_read(30)
      or die "websocket read timed out\n";
    my $read = sysread($socket, my $chunk, $length - length($buffer));
    defined $read or die "websocket read failed: $!\n";
    $read > 0 or die "websocket closed unexpectedly\n";
    $buffer .= $chunk;
  }
  return $buffer;
}

sub websocket_send {
  my ($socket, $opcode, $payload) = @_;
  my $length = length($payload);
  my $header = pack('C', 0x80 | $opcode);
  if ($length < 126) {
    $header .= pack('C', 0x80 | $length);
  } elsif ($length <= 0xffff) {
    $header .= pack('C n', 0x80 | 126, $length);
  } else {
    my $high = int($length / 4294967296);
    my $low = $length % 4294967296;
    $header .= pack('C N N', 0x80 | 127, $high, $low);
  }
  my $mask = pack('N', int(rand(4294967296)));
  my $masked = $payload;
  for my $index (0 .. $length - 1) {
    substr($masked, $index, 1) = chr(
      ord(substr($masked, $index, 1)) ^ ord(substr($mask, $index % 4, 1)));
  }
  print {$socket} $header, $mask, $masked
    or die "websocket write failed: $!\n";
}

sub websocket_receive {
  my ($socket) = @_;
  my $message = '';
  my $message_opcode;
  while (1) {
    my ($first, $second) = unpack('C C', read_exact($socket, 2));
    my $finished = ($first & 0x80) != 0;
    my $opcode = $first & 0x0f;
    my $masked = ($second & 0x80) != 0;
    my $length = $second & 0x7f;
    if ($length == 126) {
      $length = unpack('n', read_exact($socket, 2));
    } elsif ($length == 127) {
      my ($high, $low) = unpack('N N', read_exact($socket, 8));
      $length = $high * 4294967296 + $low;
    }
    my $mask = $masked ? read_exact($socket, 4) : '';
    my $payload = read_exact($socket, $length);
    if ($masked) {
      for my $index (0 .. $length - 1) {
        substr($payload, $index, 1) = chr(
          ord(substr($payload, $index, 1)) ^ ord(substr($mask, $index % 4, 1)));
      }
    }
    if ($opcode == 0x8) {
      die "websocket peer closed the connection\n";
    } elsif ($opcode == 0x9) {
      websocket_send($socket, 0xA, $payload);
      next;
    } elsif ($opcode == 0x1 || $opcode == 0x2) {
      $message = $payload;
      $message_opcode = $opcode;
    } elsif ($opcode == 0x0) {
      defined $message_opcode or die "unexpected websocket continuation\n";
      $message .= $payload;
    } else {
      next;
    }
    return $message if $finished;
  }
}

sub cdp_call {
  my ($socket, $next_id, $method, $params) = @_;
  my $id = $$next_id++;
  print STDERR "[print-pdf] $method\n";
  websocket_send($socket, 0x1, encode_json({
    id => $id,
    method => $method,
    params => $params || {},
  }));
  local $SIG{ALRM} = sub { die "$method timed out waiting for DevTools\n" };
  alarm 10;
  while (1) {
    my $response = decode_json(websocket_receive($socket));
    next unless defined $response->{id} && $response->{id} == $id;
    if (exists $response->{error}) {
      die "$method failed: " . encode_json($response->{error}) . "\n";
    }
    alarm 0;
    return $response->{result};
  }
}

sub http_json {
  my ($port, $path) = @_;
  my $socket = IO::Socket::INET->new(
    PeerAddr => '127.0.0.1',
    PeerPort => $port,
    Proto => 'tcp',
    Timeout => 1,
  ) or return;
  $socket->autoflush(1);
  print {$socket} "GET $path HTTP/1.1\r\nHost: 127.0.0.1:$port\r\nConnection: close\r\n\r\n";
  my $select = IO::Select->new($socket);
  my $response = '';
  my $content_length;
  my $header_length;
  my $deadline = time + 1;
  while (time < $deadline && $select->can_read(0.1)) {
    my $read = sysread($socket, my $chunk, 8192);
    last unless defined $read && $read > 0;
    $response .= $chunk;
    if (!defined $header_length && $response =~ /\r\n\r\n/) {
      $header_length = $+[0];
      if (substr($response, 0, $header_length) =~ /\r\nContent-Length:\s*(\d+)\r\n/i) {
        $content_length = $1;
      }
    }
    last if defined $content_length &&
      length($response) >= $header_length + $content_length;
  }
  close $socket;
  return unless defined $response && $response =~ /\AHTTP\/1\.1 200\b/;
  my (undef, $body) = split(/\r\n\r\n/, $response, 2);
  return decode_json($body);
}

my $probe = IO::Socket::INET->new(
  LocalAddr => '127.0.0.1',
  LocalPort => 0,
  Listen => 1,
  Proto => 'tcp',
  ReuseAddr => 1,
) or die "cannot reserve a DevTools port: $!\n";
my $port = $probe->sockport;
close $probe;
print STDERR "[print-pdf] launching Chromium on DevTools port $port\n";

my $url = "file://$html_path?print-pdf";
my $chrome_pid = fork();
defined $chrome_pid or die "cannot fork Chromium: $!\n";
if ($chrome_pid == 0) {
  open STDOUT, '>', '/dev/null' or die "cannot redirect Chromium stdout: $!\n";
  exec $chrome,
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--no-sandbox',
    '--host-resolver-rules=MAP * ~NOTFOUND',
    "--remote-debugging-port=$port",
    '--remote-allow-origins=*',
    "--user-data-dir=$profile",
    $url;
  die "cannot exec $chrome: $!\n";
}

my $finished = 0;
sub stop_chrome {
  return if $finished;
  kill 'TERM', $chrome_pid;
  for (1 .. 50) {
    my $waited = waitpid($chrome_pid, WNOHANG);
    if ($waited == $chrome_pid) {
      $finished = 1;
      return;
    }
    sleep 0.02;
  }
  kill 'KILL', $chrome_pid;
  waitpid($chrome_pid, 0);
  $finished = 1;
}
$SIG{INT} = sub { stop_chrome(); exit 130 };
$SIG{TERM} = sub { stop_chrome(); exit 143 };

eval {
  my $targets;
  my $deadline = time + 15;
  while (time < $deadline) {
    $targets = http_json($port, '/json/list');
    last if $targets && ref($targets) eq 'ARRAY' && @$targets;
    sleep 0.05;
  }
  $targets && ref($targets) eq 'ARRAY'
    or die "Chromium DevTools endpoint did not become ready\n";
  my ($target) = grep {
    ($_->{type} || '') eq 'page' && ($_->{url} || '') =~ /\Q$html_path\E/
  } @$targets;
  $target ||= (grep { ($_->{type} || '') eq 'page' } @$targets)[0];
  $target && $target->{webSocketDebuggerUrl}
    or die "Chromium page target was not found\n";
  print STDERR "[print-pdf] page target ready\n";
  my ($host, $ws_port, $ws_path) =
    $target->{webSocketDebuggerUrl} =~ m{\Aws://([^:/]+):(\d+)(/.*)\z};
  defined $ws_path or die "unexpected DevTools websocket URL\n";

  my $socket = IO::Socket::INET->new(
    PeerAddr => $host,
    PeerPort => $ws_port,
    Proto => 'tcp',
    Timeout => 30,
  ) or die "cannot connect to DevTools websocket: $!\n";
  $socket->autoflush(1);
  binmode $socket;
  my $key = encode_base64(pack('N4', map { int(rand(4294967296)) } 1 .. 4), '');
  print {$socket}
    "GET $ws_path HTTP/1.1\r\n",
    "Host: $host:$ws_port\r\n",
    "Upgrade: websocket\r\n",
    "Connection: Upgrade\r\n",
    "Sec-WebSocket-Key: $key\r\n",
    "Sec-WebSocket-Version: 13\r\n\r\n";
  my $headers = '';
  while ($headers !~ /\r\n\r\n/) {
    $headers .= read_exact($socket, 1);
    length($headers) < 32768 or die "oversized websocket handshake\n";
  }
  $headers =~ /\AHTTP\/1\.1 101\b/
    or die "DevTools websocket upgrade failed: $headers\n";
  print STDERR "[print-pdf] websocket ready\n";

  my $next_id = 1;
  cdp_call($socket, \$next_id, 'Page.enable', {});
  print STDERR "[print-pdf] Page domain enabled\n";
  my $layout_deadline = time + 20;
  my $pages = 0;
  my $ready = 0;
  while (time < $layout_deadline) {
    my $result = cdp_call($socket, \$next_id, 'Runtime.evaluate', {
      expression => q{({pages: document.querySelectorAll('.pdf-page').length, ready: Boolean(window.Reveal && Reveal.isReady())})},
      returnByValue => JSON::PP::true,
    });
    my $value = $result->{result}{value} || {};
    $pages = $value->{pages} || 0;
    $ready = $value->{ready} || 0;
    last if $ready && $pages == $expected_pages;
    sleep 0.05;
  }
  $ready && $pages == $expected_pages
    or die "print layout was not ready: ready=$ready pages=$pages/$expected_pages\n";
  print STDERR "[print-pdf] Reveal layout ready with $pages pages\n";

  my $pdf_result = cdp_call($socket, \$next_id, 'Page.printToPDF', {
    printBackground => JSON::PP::true,
    preferCSSPageSize => JSON::PP::true,
  });
  defined $pdf_result->{data}
    or die "Page.printToPDF returned no data\n";
  open my $pdf_fh, '>:raw', $pdf_path
    or die "cannot write $pdf_path: $!\n";
  print {$pdf_fh} decode_base64($pdf_result->{data})
    or die "cannot write complete $pdf_path: $!\n";
  close $pdf_fh or die "cannot close $pdf_path: $!\n";
  print STDERR "[print-pdf] wrote $pdf_path\n";
  close $socket;
};
my $error = $@;
stop_chrome();
die $error if $error;
